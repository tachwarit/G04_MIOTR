"""
src/eval.py
Comparaison quantitative : agent RL entraîné (DQN/PPO) vs agent baseline
à règles, sur le même protocole d'évaluation (n_episodes déterministes).
"""
import os
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


def main():
    rul_cfg = load_config("configs/rul_model.yaml")
    env_cfg = load_config("configs/env.yaml")
    train_cfg = load_config("configs/train.yaml")
    agent_cfg = load_config("configs/rule_agent.yaml")

    algo_name = train_cfg["algo"].lower()
    AlgoClass = ALGOS[algo_name]

    data = build_data_with_rul_pred(rul_cfg)
    n_episodes = agent_cfg["n_episodes"]

    # Agent à règles (baseline)
    rule_env = make_env(data, env_cfg, seed=123)
    rule_agent = RuleBasedAgent(
        maint_threshold=agent_cfg["maint_threshold"],
        critical_threshold=agent_cfg["critical_threshold"],
        rul_cap=agent_cfg["rul_cap"],
    )
    rule_stats = evaluate(rule_env, rule_agent, n_episodes=n_episodes)
    rule_stats["policy"] = "rule_based"

    # Agent RL entraîné (modèle final sauvegardé par train.py)
    model_path = os.path.join("results", f"{algo_name}_model")
    model = AlgoClass.load(model_path)
    rl_env = make_env(data, env_cfg, seed=123)
    rl_agent = SB3AgentWrapper(model)
    rl_stats = evaluate(rl_env, rl_agent, n_episodes=n_episodes)
    rl_stats["policy"] = algo_name

    comparison = pd.DataFrame([rule_stats, rl_stats])
    comparison = comparison[["policy", "mean_reward", "std_reward",
                              "n_failures", "n_maintenance", "n_stop", "n_episodes"]]
    print(comparison.to_string(index=False))

    os.makedirs("results", exist_ok=True)
    comparison.to_csv("results/comparison_rl_vs_rule.csv", index=False)

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.bar(comparison["policy"], comparison["mean_reward"],
            yerr=comparison["std_reward"], capsize=5)
    plt.ylabel("Récompense moyenne")
    plt.title(f"Agent à règles vs {algo_name.upper()} ({n_episodes} épisodes)")
    plt.tight_layout()
    plt.savefig("figures/comparison_rl_vs_rule.pdf")

    print("Comparaison sauvegardée dans results/comparison_rl_vs_rule.csv "
          "et figures/comparison_rl_vs_rule.pdf")


if __name__ == "__main__":
    main()