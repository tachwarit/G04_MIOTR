"""
src/eval.py
Comparaison quantitative : agent à règles (baseline) vs DQN vs PPO,
sur le protocole officiel à 3 seeds (configs/train.yaml -> seeds).
Remplace l'ancienne évaluation à seed unique (123), désormais périmée.
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO

from rule_agent import RuleBasedAgent, evaluate
from train import build_data_with_rul_pred, make_env
from config import load_config

ALGOS = {"dqn": DQN, "ppo": PPO}


class SB3AgentWrapper:
    """Adapte un modèle Stable-Baselines3 à l'interface .act(obs) de evaluate()."""

    def __init__(self, model):
        self.model = model

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def iqm(values):
    values = np.asarray(values, dtype=float)
    q25, q75 = np.percentile(values, [25, 75])
    mask = (values >= q25) & (values <= q75)
    return float(values[mask].mean()) if mask.any() else float(values.mean())


def aggregate(per_seed_means):
    m = float(np.mean(per_seed_means))
    if len(per_seed_means) > 1:
        sd = float(np.std(per_seed_means, ddof=1))
        se = sd / math.sqrt(len(per_seed_means))
    else:
        se = 0.0
    return m, 1.96 * se, iqm(per_seed_means)


def main():
    rul_cfg = load_config("configs/rul_model.yaml")
    env_cfg = load_config("configs/env.yaml")
    train_cfg = load_config("configs/train.yaml")
    agent_cfg = load_config("configs/rule_agent.yaml")

    seeds = train_cfg["seeds"]
    n_episodes = agent_cfg["n_episodes"]
    data = build_data_with_rul_pred(rul_cfg)

    rows = []

    # --- Baseline (agent à règles), sur les 3 seeds officiels ---
    rule_agent = RuleBasedAgent(
        maint_threshold=agent_cfg["maint_threshold"],
        critical_threshold=agent_cfg["critical_threshold"],
        rul_cap=agent_cfg["rul_cap"],
    )
    baseline_means, baseline_failures = [], []
    for seed in seeds:
        env = make_env(data, env_cfg, seed=seed)
        stats = evaluate(env, rule_agent, n_episodes=n_episodes)
        baseline_means.append(stats["mean_reward"])
        baseline_failures.append(stats["n_failures"])
        rows.append({"policy": "rule_based", "seed": seed,
                      "mean_reward": stats["mean_reward"], "n_failures": stats["n_failures"]})
    m, ci, q = aggregate(baseline_means)
    rows.append({"policy": "rule_based", "seed": "aggregate", "mean_reward": m,
                 "ci95": ci, "iqm": q, "n_failures": sum(baseline_failures)})

    # --- DQN / PPO, modèles sauvegardés par seed (results/{algo}_seed{n}_model.zip) ---
    for algo_name, AlgoClass in ALGOS.items():
        algo_means, algo_failures = [], []
        for seed in seeds:
            model_path = os.path.join("results", f"{algo_name}_seed{seed}_model")
            if not os.path.exists(model_path + ".zip"):
                print(f"[manquant] {model_path}.zip — lance python src/train.py d'abord")
                continue
            model = AlgoClass.load(model_path)
            env = make_env(data, env_cfg, seed=seed)
            stats = evaluate(env, SB3AgentWrapper(model), n_episodes=n_episodes)
            algo_means.append(stats["mean_reward"])
            algo_failures.append(stats["n_failures"])
            rows.append({"policy": algo_name, "seed": seed,
                          "mean_reward": stats["mean_reward"], "n_failures": stats["n_failures"]})
        if algo_means:
            m, ci, q = aggregate(algo_means)
            rows.append({"policy": algo_name, "seed": "aggregate", "mean_reward": m,
                         "ci95": ci, "iqm": q, "n_failures": sum(algo_failures)})

    results = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/comparison_rl_vs_rule.csv", index=False)
    print(results.to_string(index=False))

    # --- Figure : moyenne agrégée ± IC95, sur les 3 politiques ---
    agg = results[results.seed == "aggregate"].set_index("policy")
    policies = [p for p in ["rule_based", "dqn", "ppo"] if p in agg.index]
    means = [agg.loc[p, "mean_reward"] for p in policies]
    cis = [agg.loc[p, "ci95"] for p in policies]

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["#4C72B0", "#55A868", "#C44E52"][:len(policies)]
    ax.bar(policies, means, yerr=cis, capsize=6, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Récompense moyenne (± IC 95%, approximation normale)")
    ax.set_title(f"Comparaison des politiques ({len(seeds)} seeds officiels)")
    plt.tight_layout()
    plt.savefig("figures/comparison_rl_vs_rule.pdf")

    print("\nSauvegardé dans results/comparison_rl_vs_rule.csv "
          "et figures/comparison_rl_vs_rule.pdf")


if __name__ == "__main__":
    main()
