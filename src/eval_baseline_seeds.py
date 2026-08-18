"""
src/eval_baseline_seeds.py
Évalue l'agent à règles sur les seeds officiels du protocole (configs/train.yaml
-> seeds), pour remplir le Tableau 3 (Internal Reference / Baseline) du rapport.
"""
import os
import pandas as pd

from rule_agent import RuleBasedAgent, evaluate
from train import build_data_with_rul_pred, make_env
from config import load_config


def main():
    rul_cfg = load_config("configs/rul_model.yaml")
    env_cfg = load_config("configs/env.yaml")
    agent_cfg = load_config("configs/rule_agent.yaml")
    train_cfg = load_config("configs/train.yaml")

    seeds = train_cfg["seeds"]  # {0, 1, 2} - mêmes seeds officiels que DQN/PPO
    n_episodes = agent_cfg["n_episodes"]

    data = build_data_with_rul_pred(rul_cfg)
    agent = RuleBasedAgent(
        maint_threshold=agent_cfg["maint_threshold"],
        critical_threshold=agent_cfg["critical_threshold"],
        rul_cap=agent_cfg["rul_cap"],
    )

    rows = []
    for seed in seeds:
        env = make_env(data, env_cfg, seed=seed)
        stats = evaluate(env, agent, n_episodes=n_episodes)
        stats["seed"] = seed
        rows.append(stats)
        print(f"seed={seed}  mean_reward={stats['mean_reward']:.3f}  "
              f"std={stats['std_reward']:.3f}  failures={stats['n_failures']}")

    results = pd.DataFrame(rows)
    results = results[["seed", "mean_reward", "std_reward",
                        "n_failures", "n_maintenance", "n_stop", "n_episodes"]]

    mean_row = results.drop(columns="seed").mean(numeric_only=True)
    mean_row["seed"] = "mean"
    results = pd.concat([results, pd.DataFrame([mean_row])], ignore_index=True)

    print("\n" + results.to_string(index=False))

    os.makedirs("results", exist_ok=True)
    results.to_csv("results/baseline_seed_sweep.csv", index=False)
    print("\nSauvegardé dans results/baseline_seed_sweep.csv")


if __name__ == "__main__":
    main()
