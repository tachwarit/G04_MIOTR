"""
src/train.py
Entraînement de l'agent RL (DQN ou PPO) sur l'environnement de maintenance
prédictive personnalisé (PredictiveMaintenanceEnv).
"""
import os
import pandas as pd
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


def train_one(algo_name, AlgoClass, data, env_cfg, train_cfg, seed, save_model=False):
    log_dir = os.path.join("logs", algo_name, f"seed_{seed}")
    train_env = make_env(data, env_cfg, seed=seed, log_dir=log_dir)
    eval_env = make_env(data, env_cfg, seed=seed + 1000)

    model = AlgoClass(
        "MlpPolicy",
        train_env,
        learning_rate=train_cfg["learning_rate"],
        seed=seed,
        verbose=1,
        tensorboard_log="logs",
    )

    best_path = os.path.join("results", f"{algo_name}_best") if save_model else None
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_path,
        log_path=log_dir,
        eval_freq=train_cfg["eval_freq"],
        n_eval_episodes=20,
        deterministic=True,
    )

    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=eval_callback,
        tb_log_name=f"{algo_name}_seed{seed}",
    )

    if save_model:
        os.makedirs("results", exist_ok=True)
        model_path = os.path.join("results", f"{algo_name}_model")
        model.save(model_path)
        print(f"Modèle de référence (seed={seed}) sauvegardé dans {model_path}.zip")

    return log_dir


def build_training_curves_csv(runs):
    """Construit results/training_curves.csv (schéma : algo, seed, step,
    episode_return) à partir des monitor.csv de chaque run (algo, seed)."""
    rows = []
    for algo_name, seed, log_dir in runs:
        monitor_path = os.path.join(log_dir, "monitor.csv")
        run_df = pd.read_csv(monitor_path, skiprows=1)
        run_df["step"] = run_df["l"].cumsum()
        for _, row in run_df.iterrows():
            rows.append({
                "algo": algo_name, "seed": seed,
                "step": int(row["step"]), "episode_return": row["r"],
            })
    curves = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    curves.to_csv("results/training_curves.csv", index=False)
    print(f"Courbes d'entraînement sauvegardées dans results/training_curves.csv "
          f"({len(curves)} lignes, {curves['algo'].nunique()} algo(s), "
          f"{curves['seed'].nunique()} seed(s))")


def main():
    rul_cfg = load_config("configs/rul_model.yaml")
    env_cfg = load_config("configs/env.yaml")
    train_cfg = load_config("configs/train.yaml")

    algo_names = [a.lower() for a in train_cfg["algos"]]
    seeds = train_cfg["seeds"]
    for algo_name in algo_names:
        if algo_name not in ALGOS:
            raise ValueError(f"algos doit contenir 'dqn' et/ou 'ppo', reçu : {algo_name}")

    data = build_data_with_rul_pred(rul_cfg)

    runs = []
    for algo_name in algo_names:
        for i, seed in enumerate(seeds):
            print(f"\n===== {algo_name.upper()} - seed {seed} ({i + 1}/{len(seeds)}) =====")
            log_dir = train_one(
                algo_name, ALGOS[algo_name], data, env_cfg, train_cfg,
                seed=seed, save_model=(i == 0),
            )
            runs.append((algo_name, seed, log_dir))

    build_training_curves_csv(runs)


if __name__ == "__main__":
    main()
