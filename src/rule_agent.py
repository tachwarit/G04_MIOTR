"""
src/rule_agent.py
Agent baseline à règles (seuils fixes sur le RUL estimé) - Milestone 1.
"""
import numpy as np


class RuleBasedAgent:
    """Politique à seuils fixes sur le RUL estimé.

    - RUL estimé > maint_threshold                        -> continuer (0)
    - critical_threshold < RUL estimé <= maint_threshold   -> maintenance (1)
    - RUL estimé <= critical_threshold                     -> arrêt d'urgence (2)
    """

    def __init__(self, maint_threshold=30, critical_threshold=5, rul_cap=125):
        self.maint_threshold = maint_threshold
        self.critical_threshold = critical_threshold
        self.rul_cap = rul_cap

    def act(self, obs):
        rul_pred = obs[0] * self.rul_cap  # dénormalise obs[0] (0..1) en cycles
        if rul_pred <= self.critical_threshold:
            return 2
        if rul_pred <= self.maint_threshold:
            return 1
        return 0


def evaluate(env, agent, n_episodes=100):
    """Évalue l'agent sur n_episodes et renvoie des statistiques agrégées."""
    rewards, events = [], []
    for _ in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        terminated = False
        while not terminated:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        events.append(info["event"])
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "n_failures": events.count("unplanned_failure"),
        "n_maintenance": events.count("maintenance"),
        "n_stop": events.count("stop"),
        "n_episodes": n_episodes,
    }


if __name__ == "__main__":
    import os
    import pandas as pd
    from data import load_fd001, compute_train_rul, SETTING_NAMES
    from rul_model import select_features
    from env import PredictiveMaintenanceEnv
    from sklearn.ensemble import RandomForestRegressor

    train, _, _ = load_fd001()
    train = compute_train_rul(train)
    feature_cols = SETTING_NAMES + select_features(train)

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(train[feature_cols], train["RUL"].clip(upper=125))
    train["RUL_pred"] = rf.predict(train[feature_cols])

    env = PredictiveMaintenanceEnv(train, rul_cap=125, seed=0)
    agent = RuleBasedAgent(maint_threshold=30, critical_threshold=5, rul_cap=125)

    stats = evaluate(env, agent, n_episodes=100)
    print("Baseline agent à règles - 100 épisodes :")
    for k, v in stats.items():
        print(f"  {k} : {v}")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame([stats]).to_csv("results/rule_agent_metrics.csv", index=False)
    print("Résultats sauvegardés dans results/rule_agent_metrics.csv")
