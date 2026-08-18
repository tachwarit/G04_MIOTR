"""
src/ablation_continue_reward.py
Teste une valeur de continue_reward pour la Section VII (Ablation Study),
SANS modifier configs/env.yaml (utilisé partout ailleurs dans le pipeline).
Usage : python src/ablation_continue_reward.py 0.5
"""
import sys
from sklearn.ensemble import RandomForestRegressor

from data import load_fd001, compute_train_rul, SETTING_NAMES
from rul_model import select_features
from env import PredictiveMaintenanceEnv
from rule_agent import evaluate
from config import load_config
from stable_baselines3 import DQN


class SB3AgentWrapper:
    def __init__(self, model):
        self.model = model

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def main(continue_reward_value):
    rul_cfg = load_config("configs/rul_model.yaml")
    env_cfg = load_config("configs/env.yaml")
    train_cfg = load_config("configs/train.yaml")

    # Override en mémoire uniquement : ne touche pas au fichier YAML.
    reward_params = dict(env_cfg["reward"])
    reward_params["continue_reward"] = continue_reward_value
    print(f"Test avec continue_reward = {continue_reward_value}")

    train, _, _ = load_fd001()
    train = compute_train_rul(train)
    feature_cols = SETTING_NAMES + select_features(train, rul_cfg["variance_threshold"])

    rf = RandomForestRegressor(
        n_estimators=rul_cfg["n_estimators"], max_depth=rul_cfg["max_depth"],
        random_state=rul_cfg["random_state"], n_jobs=-1,
    )
    rf.fit(train[feature_cols], train["RUL"].clip(upper=rul_cfg["rul_cap"]))
    train["RUL_pred"] = rf.predict(train[feature_cols])

    seed = train_cfg["seeds"][0]  # seed 0, comme la ligne "pilote" du Tableau 5
    train_env = PredictiveMaintenanceEnv(
        train, rul_cap=env_cfg["rul_cap"], reward_params=reward_params, seed=seed
    )
    model = DQN(
        "MlpPolicy", train_env, learning_rate=train_cfg["learning_rate"],
        seed=seed, verbose=1,
    )
    model.learn(total_timesteps=train_cfg["total_timesteps"])

    eval_env = PredictiveMaintenanceEnv(
        train, rul_cap=env_cfg["rul_cap"], reward_params=reward_params, seed=seed
    )
    stats = evaluate(eval_env, SB3AgentWrapper(model), n_episodes=100)
    print(f"\nRÉSULTAT continue_reward={continue_reward_value} : "
          f"mean_reward={stats['mean_reward']:.3f}  "
          f"std={stats['std_reward']:.3f}  "
          f"failures={stats['n_failures']}/100")


if __name__ == "__main__":
    value = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    main(value)
