"""
src/env.py
Environnement Gymnasium custom : maintenance conditionnelle sur des
trajectoires de moteurs (NASA C-MAPSS FD001).

État (observation, Box(3,)) :
    [0] RUL estimé normalisé (0..1, plafonné à rul_cap)
    [1] cycle courant normalisé (historique d'usage, 0..1)
    [2] tendance du RUL estimé (dégradation récente, -1..1)

Actions (Discrete(3)) :
    0 = continuer l'opération
    1 = planifier une maintenance (termine l'épisode)
    2 = arrêter l'équipement (arrêt d'urgence, termine l'épisode)

Récompense : voir reward_params dans __init__.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PredictiveMaintenanceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data, rul_cap=125, reward_params=None, seed=None):
        """
        data : DataFrame avec au minimum les colonnes
               ['unit_nr', 'time_cycles', 'RUL', 'RUL_pred']
               RUL      = vérité terrain (utilisée seulement pour la récompense)
               RUL_pred = estimation du modèle Random Forest (ce que voit l'agent)
        """
        super().__init__()
        self.data = data
        self.rul_cap = rul_cap
        self.units = data["unit_nr"].unique()
        self.rng = np.random.default_rng(seed)

        params = {
            "failure_penalty": -50.0,   # panne non planifiée
            "stop_cost": -10.0,         # arrêt d'urgence (coût fixe)
            "maint_base": 15.0,         # bonus de base pour une maintenance
            "maint_waste_coef": 0.3,    # pénalité par cycle de RUL réel gaspillé
            "continue_reward": 1.0,     # récompense par cycle d'opération normale
        }
        if reward_params:
            params.update(reward_params)
        self.reward_params = params

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        self._traj = None
        self._idx = 0
        self._prev_rul_pred = None

    def _load_episode(self, unit_nr=None):
        if unit_nr is None:
            unit_nr = self.rng.choice(self.units)
        traj = self.data[self.data.unit_nr == unit_nr].sort_values("time_cycles")
        return traj.reset_index(drop=True)

    def _obs(self):
        row = self._traj.iloc[self._idx]
        rul_pred_norm = min(row.RUL_pred, self.rul_cap) / self.rul_cap
        cycle_norm = min(row.time_cycles / self.rul_cap, 1.0)
        trend = 0.0 if self._prev_rul_pred is None else float(
            np.clip((row.RUL_pred - self._prev_rul_pred) / self.rul_cap, -1.0, 1.0)
        )
        return np.array([rul_pred_norm, cycle_norm, trend], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        unit_nr = options.get("unit_nr") if options else None
        self._traj = self._load_episode(unit_nr)
        self._idx = 0
        self._prev_rul_pred = None
        obs = self._obs()
        self._prev_rul_pred = self._traj.iloc[0].RUL_pred
        return obs, {"unit_nr": int(self._traj.iloc[0].unit_nr)}

    def step(self, action):
        row = self._traj.iloc[self._idx]
        true_rul = row.RUL
        p = self.reward_params
        terminated = False
        info = {"unit_nr": int(row.unit_nr), "cycle": int(row.time_cycles),
                 "true_rul": int(true_rul)}

        if action == 0:  # continuer
            if true_rul <= 0 or self._idx == len(self._traj) - 1:
                reward = p["failure_penalty"]
                terminated = True
                info["event"] = "unplanned_failure"
            else:
                reward = p["continue_reward"]
                self._idx += 1
        elif action == 1:  # planifier une maintenance
            reward = p["maint_base"] - p["maint_waste_coef"] * true_rul
            terminated = True
            info["event"] = "maintenance"
        elif action == 2:  # arrêter l'équipement
            reward = p["stop_cost"]
            terminated = True
            info["event"] = "stop"
        else:
            raise ValueError(f"Action invalide : {action}")

        if not terminated:
            obs = self._obs()
            self._prev_rul_pred = self._traj.iloc[self._idx].RUL_pred
        else:
            obs = np.zeros(3, dtype=np.float32)

        return obs, reward, terminated, False, info


if __name__ == "__main__":
    # Test rapide : politique aléatoire sur des trajectoires réelles,
    # avec RUL_pred généré par le modèle Random Forest de rul_model.py
    from data import load_fd001, compute_train_rul, SETTING_NAMES
    from rul_model import select_features
    from config import load_config
    from sklearn.ensemble import RandomForestRegressor

    rul_cfg = load_config("configs/rul_model.yaml")
    env_cfg = load_config("configs/env.yaml")

    train, _, _ = load_fd001()
    train = compute_train_rul(train)
    feature_cols = SETTING_NAMES + select_features(train, rul_cfg["variance_threshold"])

    rf = RandomForestRegressor(
        n_estimators=rul_cfg["n_estimators"], max_depth=rul_cfg["max_depth"],
        random_state=rul_cfg["random_state"], n_jobs=-1,
    )
    rf.fit(train[feature_cols], train["RUL"].clip(upper=rul_cfg["rul_cap"]))
    train["RUL_pred"] = rf.predict(train[feature_cols])

    env = PredictiveMaintenanceEnv(
        train, rul_cap=env_cfg["rul_cap"], reward_params=env_cfg["reward"], seed=0
    )
    obs, info = env.reset()
    total_reward = 0.0
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            print(f"moteur {info['unit_nr']:>3} | événement={info['event']:<17} "
                  f"| cycle={info['cycle']:>3} | reward cumulé={total_reward:.1f}")
            obs, info = env.reset()
            total_reward = 0.0
    print("Test avec politique aléatoire terminé.")
