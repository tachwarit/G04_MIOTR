"""
src/robustness_test.py
Test de robustesse : stabilité des politiques (règles, DQN, PPO) face à
différents profils de dégradation (moteurs à vie courte / moyenne / longue).
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rule_agent import RuleBasedAgent, evaluate
from train import build_data_with_rul_pred, make_env
from eval import SB3AgentWrapper, ALGOS
from config import load_config


def split_by_life_profile(data):
    """Découpe les moteurs en 3 profils selon leur durée de vie totale (tertiles) :
    courte (dégradation rapide), moyenne, longue (dégradation lente)."""
    life = data.groupby("unit_nr")["time_cycles"].max()
    q1, q2 = life.quantile([1 / 3, 2 / 3])
    profiles = {
        "courte": life[life <= q1].index,
        "moyenne": life[(life > q1) & (life <= q2)].index,
        "longue": life[life > q2].index,
    }
    for name, units in profiles.items():
        print(f"Profil '{name}' : {len(units)} moteurs "
              f"(vie {life[units].min()}-{life[units].max()} cycles)")
    return profiles


def build_agents(agent_cfg):
    """Construit toutes les politiques disponibles : règles + modèles RL entraînés."""
    agents = {"rule_based": RuleBasedAgent(
        maint_threshold=agent_cfg["maint_threshold"],
        critical_threshold=agent_cfg["critical_threshold"],
        rul_cap=agent_cfg["rul_cap"],
    )}
    for algo_name, AlgoClass in ALGOS.items():
        model_path = os.path.join("results", f"{algo_name}_model")
        if os.path.exists(model_path + ".zip"):
            agents[algo_name] = SB3AgentWrapper(AlgoClass.load(model_path))
    return agents


def main():
    rul_cfg = load_config("configs/rul_model.yaml")
    env_cfg = load_config("configs/env.yaml")
    agent_cfg = load_config("configs/rule_agent.yaml")

    data = build_data_with_rul_pred(rul_cfg)
    profiles = split_by_life_profile(data)
    agents = build_agents(agent_cfg)
    n_episodes = agent_cfg["n_episodes"]

    rows = []
    for profile_name, units in profiles.items():
        subset = data[data.unit_nr.isin(units)]
        for policy_name, agent in agents.items():
            env = make_env(subset, env_cfg, seed=123)
            stats = evaluate(env, agent, n_episodes=n_episodes)
            stats["profile"] = profile_name
            stats["policy"] = policy_name
            rows.append(stats)

    results = pd.DataFrame(rows)
    results = results[["profile", "policy", "mean_reward", "std_reward",
                        "n_failures", "n_maintenance", "n_stop", "n_episodes"]]
    print(results.to_string(index=False))

    os.makedirs("results", exist_ok=True)
    results.to_csv("results/robustness_test.csv", index=False)

    os.makedirs("figures", exist_ok=True)
    pivot = results.pivot(index="profile", columns="policy", values="mean_reward")
    pivot = pivot.reindex(["courte", "moyenne", "longue"])
    pivot.plot(kind="bar", figsize=(7, 5))
    plt.ylabel("Récompense moyenne")
    plt.title("Robustesse par profil de dégradation")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("figures/robustness_test.pdf")

    print("Résultats sauvegardés dans results/robustness_test.csv "
          "et figures/robustness_test.pdf")


if __name__ == "__main__":
    main()
