"""
src/make_comparison_video.py
Génère videos/before_after_comparison.mp4 : deux panneaux côte à côte
montrant le RUL estimé au fil des cycles pour (1) une politique aléatoire
("avant entraînement") et (2) l'agent DQN entraîné ("après entraînement"),
sur la même trajectoire de moteur, avec un marqueur à l'événement final
(maintenance / arrêt / panne).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
from stable_baselines3 import DQN

from train import build_data_with_rul_pred, make_env
from config import load_config

UNIT_NR = 1  # même moteur utilisé pour fd001_unit1_sensors.pdf, pour la cohérence
ACTION_NAMES = {0: "continuer", 1: "maintenance", 2: "arrêt"}
EVENT_COLOR = {"unplanned_failure": "red", "maintenance": "green", "stop": "orange"}


def run_episode(env, act_fn, unit_nr, seed=None):
    obs, info = env.reset(seed=seed, options={"unit_nr": unit_nr})
    cycles, ruls = [], []
    final_event = None
    terminated = False
    while not terminated:
        cycles.append(int(env._traj.iloc[env._idx].time_cycles))
        ruls.append(float(env._traj.iloc[env._idx].RUL_pred))
        action = act_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            final_event = (action, info.get("event", "?"))
    return cycles, ruls, final_event


def run_random_episode_illustrative(env, unit_nr, min_length=15, max_attempts=30):
    """La politique aléatoire a ~2/3 de chances de s'arrêter dès le 1er cycle
    (1 action sur 3 = continuer). On réessaie plusieurs tirages pour trouver
    un exemple assez long à montrer, en gardant le plus long si aucun
    n'atteint min_length. La politique reste 100% aléatoire à chaque essai,
    seul l'exemple retenu pour la vidéo est choisi pour être illustratif."""
    best = None
    for attempt in range(max_attempts):
        env.action_space.seed(attempt)
        random_policy = lambda obs: env.action_space.sample()
        cycles, ruls, event = run_episode(env, random_policy, unit_nr, seed=attempt)
        if best is None or len(cycles) > len(best[0]):
            best = (cycles, ruls, event)
        if len(cycles) >= min_length:
            print(f"Episode aléatoire illustratif trouvé à l'essai {attempt + 1} "
                  f"({len(cycles)} cycles)")
            return best
    print(f"Aucun essai n'a dépassé {min_length} cycles ; "
          f"on garde le plus long trouvé ({len(best[0])} cycles).")
    return best


def main():
    rul_cfg = load_config("configs/rul_model.yaml")
    env_cfg = load_config("configs/env.yaml")

    data = build_data_with_rul_pred(rul_cfg)

    env_before = make_env(data, env_cfg, seed=0)
    env_after = make_env(data, env_cfg, seed=0)

    random_policy_note = ("Politique aléatoire : ~2/3 de chances de s'arrêter dès "
                           "le 1er cycle (2 actions sur 3 terminent l'épisode). "
                           "On garde ici un tirage assez long pour être lisible ; "
                           "la politique elle-même reste entièrement aléatoire.")
    print(random_policy_note)
    cycles_b, ruls_b, event_b = run_random_episode_illustrative(env_before, UNIT_NR)

    model = DQN.load("results/dqn_model")
    trained_policy = lambda obs: int(model.predict(obs, deterministic=True)[0])
    cycles_a, ruls_a, event_a = run_episode(env_after, trained_policy, UNIT_NR)
    print(f"Avant (aléatoire) : {len(cycles_b)} cycles, événement={event_b}")
    print(f"Après (DQN)       : {len(cycles_a)} cycles, événement={event_a}")

    n_frames = max(len(cycles_b), len(cycles_a)) + 10
    x_max = max(cycles_b[-1] if cycles_b else 1, cycles_a[-1] if cycles_a else 1) + 5

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, title in zip(axes, ["Avant entraînement (politique aléatoire)",
                                 "Après entraînement (DQN)"]):
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 130)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("RUL estimé")
        ax.set_title(title, fontsize=10)

    line_b, = axes[0].plot([], [], color="#4C72B0", linewidth=2)
    line_a, = axes[1].plot([], [], color="#4C72B0", linewidth=2)
    label_b = axes[0].text(0.05, 0.92, "", transform=axes[0].transAxes)
    label_a = axes[1].text(0.05, 0.92, "", transform=axes[1].transAxes)

    frames = []
    for f in range(n_frames):
        ib = min(f, len(cycles_b) - 1)
        ia = min(f, len(cycles_a) - 1)
        line_b.set_data(cycles_b[:ib + 1], ruls_b[:ib + 1])
        line_a.set_data(cycles_a[:ia + 1], ruls_a[:ia + 1])

        if f >= len(cycles_b) - 1 and event_b:
            action, event = event_b
            axes[0].scatter([cycles_b[-1]], [ruls_b[-1]], s=120,
                             color=EVENT_COLOR.get(event, "gray"), zorder=5)
            label_b.set_text(f"{ACTION_NAMES[action]} — {event}")

        if f >= len(cycles_a) - 1 and event_a:
            action, event = event_a
            axes[1].scatter([cycles_a[-1]], [ruls_a[-1]], s=120,
                             color=EVENT_COLOR.get(event, "gray"), zorder=5)
            label_a.set_text(f"{ACTION_NAMES[action]} — {event}")

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(img)

    os.makedirs("videos", exist_ok=True)
    out_path = "videos/before_after_comparison.mp4"
    imageio.mimsave(out_path, frames, fps=10)
    print(f"Vidéo sauvegardée dans {out_path}")


if __name__ == "__main__":
    main()