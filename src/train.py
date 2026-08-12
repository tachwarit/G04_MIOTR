"""
src/train.py
Entraînement de l'agent RL (DQN ou PPO) sur l'environnement de maintenance
prédictive personnalisé (PredictiveMaintenanceEnv).
"""
import os
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from sklearn.ensemble import RandomForestRegressor

from data import load_fd001, compute_train_rul, SETTING_NAMES
from rul_model import select_features
from env import PredictiveMaintenanceEnv
from config import load_config

ALGOS = {"dqn": DQN, "ppo": PPO}


def build_data_with_rul_pred(rul_cfg):
    """Charge FD001 et ajoute la colonne RUL_pred (modèle Random Forest)."""
    train, _, _ = load_fd001()
    train = compute_train_rul(train)
    feature_cols = SETTING_NAMES + select_features(train, rul_cfg["variance_threshold"])

    rf = RandomForestRegressor(
        n_estimators=rul_cfg["n_estimators"], max_depth=rul_cfg["max_depth"],
        random_state=rul_cfg["random_state"], n_jobs=-1,
    )
    rf.fit(train[feature_cols], train["RUL"].clip(upper=rul_cfg["rul_cap"]))
    train["RUL_pred"] = rf.predict(train[feature_cols])
    return train


def make_env(data, env_cfg, seed=None, log_dir=None):
    env = PredictiveMaintenanceEnv(
        data, rul_cap=env_cfg["rul_cap"], reward_params=env_cfg["reward"], seed=seed
    )
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
    return env


def train_one(algo_name, AlgoClass, data, env_cfg, train_cfg):
    log_dir = os.path.join("logs", algo_name)
    train_env = make_env(data, env_cfg, seed=train_cfg["seed"], log_dir=log_dir)
    eval_env = make_env(data, env_cfg, seed=train_cfg["seed"] + 1000)

    model = AlgoClass(
        "MlpPolicy",
        train_env,
        learning_rate=train_cfg["learning_rate"],
        seed=train_cfg["seed"],
        verbose=1,
        tensorboard_log="logs",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join("results", f"{algo_name}_best"),
        log_path=log_dir,
        eval_freq=train_cfg["eval_freq"],
        n_eval_episodes=20,
        deterministic=True,
    )

    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=eval_callback,
        tb_log_name=algo_name,
    )

    os.makedirs("results", exist_ok=True)
    model_path = os.path.join("results", f"{algo_name}_model")
    model.save(model_path)
    print(f"Modèle entraîné sauvegardé dans {model_path}.zip")
    print(f"Logs TensorBoard dans logs/{algo_name}_1/ "
          f"(lance : tensorboard --logdir logs)")


def main():
    rul_cfg = load_config("configs/rul_model.yaml")
    env_cfg = load_config("configs/env.yaml")
    train_cfg = load_config("configs/train.yaml")

    algo_names = [a.lower() for a in train_cfg["algos"]]
    for algo_name in algo_names:
        if algo_name not in ALGOS:
            raise ValueError(f"algos doit contenir 'dqn' et/ou 'ppo', reçu : {algo_name}")

    data = build_data_with_rul_pred(rul_cfg)

    for algo_name in algo_names:
        print(f"\n===== Entraînement {algo_name.upper()} =====")
        train_one(algo_name, ALGOS[algo_name], data, env_cfg, train_cfg)


if __name__ == "__main__":
    main()
