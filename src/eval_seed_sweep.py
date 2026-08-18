"""
src/eval_seed_sweep.py
Évalue chaque modèle DQN/PPO sauvegardé par seed (results/{algo}_seed{n}_model.zip),
calcule l'IQM et une CI (normale, interim) sur les seeds, et lit dans les logs
d'évaluation le nombre de pas nécessaires pour dépasser la baseline
(results/baseline_seed_sweep.csv, moyenne poolée).
À lancer APRES avoir relancé python src/train.py avec la version corrigée
qui sauvegarde un modèle par seed.
"""
import os
import math
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO

from rule_agent import evaluate
from train import build_data_with_rul_pred, make_env
from eval import SB3AgentWrapper
from config import load_config

ALGOS = {"dqn": DQN, "ppo": PPO}


def steps_to_threshold(log_dir, threshold):
    """Premier pas d'entraînement où l'évaluation déterministe (EvalCallback)
    atteint ou dépasse `threshold`. None si jamais atteint."""
    path = os.path.join(log_dir, "evaluations.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    means = data["results"].mean(axis=1)
    hits = np.where(means >= threshold)[0]
    return int(data["timesteps"][hits[0]]) if len(hits) else None


def iqm(values):
    values = np.asarray(values, dtype=float)
    q25, q75 = np.percentile(values, [25, 75])
    mask = (values >= q25) & (values <= q75)
    return float(values[mask].mean()) if mask.any() else float(values.mean())


def main():
    rul_cfg = load_config("configs/rul_model.yaml")
    env_cfg = load_config("configs/env.yaml")
    train_cfg = load_config("configs/train.yaml")
    agent_cfg = load_config("configs/rule_agent.yaml")

    seeds = train_cfg["seeds"]
    n_episodes = agent_cfg["n_episodes"]

    baseline_mean = pd.read_csv("results/baseline_seed_sweep.csv")
    baseline_mean = baseline_mean.loc[baseline_mean.seed == "mean", "mean_reward"].iloc[0]
    print(f"Seuil (moyenne baseline poolée) = {baseline_mean:.3f}")

    data = build_data_with_rul_pred(rul_cfg)

    rows = []
    for algo_name, AlgoClass in ALGOS.items():
        per_seed_returns = []
        for seed in seeds:
            model_path = os.path.join("results", f"{algo_name}_seed{seed}_model")
            if not os.path.exists(model_path + ".zip"):
                print(f"[manquant] {model_path}.zip — relance python src/train.py d'abord")
                continue
            model = AlgoClass.load(model_path)
            env = make_env(data, env_cfg, seed=seed)
            stats = evaluate(env, SB3AgentWrapper(model), n_episodes=n_episodes)
            log_dir = os.path.join("logs", algo_name, f"seed_{seed}")
            steps = steps_to_threshold(log_dir, baseline_mean)
            per_seed_returns.append(stats["mean_reward"])
            rows.append({"algo": algo_name, "seed": seed,
                         "mean_reward": stats["mean_reward"],
                         "n_failures": stats["n_failures"],
                         "steps_to_threshold": steps})
            print(f"{algo_name} seed={seed}: mean_reward={stats['mean_reward']:.3f} "
                  f"failures={stats['n_failures']} steps_to_threshold={steps}")

        if per_seed_returns:
            m = np.mean(per_seed_returns)
            sd = np.std(per_seed_returns, ddof=1) if len(per_seed_returns) > 1 else 0.0
            se = sd / math.sqrt(len(per_seed_returns))
            print(f"{algo_name} AGGREGATE: mean={m:.3f} ±{1.96*se:.3f}  "
                  f"IQM={iqm(per_seed_returns):.3f}")

    results = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/seed_sweep_eval.csv", index=False)
    print("\nSauvegardé dans results/seed_sweep_eval.csv")


if __name__ == "__main__":
    main()
