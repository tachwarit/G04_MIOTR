#!/usr/bin/env python3
"""Génère une vidéo de démonstration pour comparer un agent à règles
et un agent RL entraîné sur l'environnement de maintenance prédictive.

Usage :
    python src/generate_video.py --algo ppo --unit-nr 1 --output videos/ppo_vs_rule.mp4
    python src/generate_video.py --algo all --output videos/comparison.mp4
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import imageio

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    from train import build_data_with_rul_pred
    from rule_agent import RuleBasedAgent
    from env import PredictiveMaintenanceEnv
    from config import load_config
except ImportError:  # pragma: no cover
    from src.train import build_data_with_rul_pred
    from src.rule_agent import RuleBasedAgent
    from src.env import PredictiveMaintenanceEnv
    from src.config import load_config

from stable_baselines3 import DQN, PPO


ACTION_LABELS = {0: "continuer", 1: "maintenance", 2: "arrêt"}
ACTION_COLORS = {0: "#4C72B0", 1: "#55A868", 2: "#C44E52"}


class SB3AgentWrapper:
    """Adapte un modèle Stable-Baselines3 à l'interface .act(obs)."""

    def __init__(self, model):
        self.model = model

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def load_rule_agent():
    agent_cfg = load_config(str(ROOT / "configs" / "rule_agent.yaml"))
    return RuleBasedAgent(
        maint_threshold=agent_cfg["maint_threshold"],
        critical_threshold=agent_cfg["critical_threshold"],
        rul_cap=agent_cfg["rul_cap"],
    )


def load_rl_agent(algo_name):
    train_cfg = load_config(str(ROOT / "configs" / "train.yaml"))
    seeds = train_cfg.get("seeds", [0])
    model_candidates = []
    for seed in seeds:
        model_candidates.append(ROOT / "results" / f"{algo_name}_seed{seed}_model.zip")
    model_candidates.append(ROOT / "results" / f"{algo_name}_model.zip")

    chosen = None
    for candidate in model_candidates:
        if candidate.exists():
            chosen = candidate
            break
    if chosen is None:
        raise FileNotFoundError(
            f"Aucun modèle sauvegardé trouvé pour '{algo_name}'. "
            f"Vérifie les résultats dans {ROOT / 'results'} ou lance python src/train.py."
        )

    algo_cls = {"dqn": DQN, "ppo": PPO}[algo_name]
    model = algo_cls.load(str(chosen))
    return SB3AgentWrapper(model)


def make_env(seed=0):
    rul_cfg = load_config(str(ROOT / "configs" / "rul_model.yaml"))
    env_cfg = load_config(str(ROOT / "configs" / "env.yaml"))
    data = build_data_with_rul_pred(rul_cfg)
    env = PredictiveMaintenanceEnv(
        data,
        rul_cap=env_cfg["rul_cap"],
        reward_params=env_cfg["reward"],
        seed=seed,
    )
    return env


def run_episode(env, agent, unit_nr, max_steps=200):
    obs, info = env.reset(options={"unit_nr": int(unit_nr)})
    history = []
    total_reward = 0.0
    step = 0

    while step < max_steps:
        row = env._traj.iloc[env._idx]
        rul_pred = float(obs[0] * env.rul_cap)
        action = int(agent.act(obs))
        prev_info = {
            "unit_nr": int(row["unit_nr"]),
            "cycle": int(row["time_cycles"]),
            "true_rul": int(row["RUL"]),
        }
        obs_next, reward, terminated, _, info = env.step(action)
        total_reward += float(reward)
        history.append(
            {
                "step": step,
                "cycle": int(prev_info["cycle"]),
                "true_rul": int(prev_info["true_rul"]),
                "rul_pred": rul_pred,
                "reward": float(reward),
                "action": int(action),
                "event": info.get("event", "-"),
                "terminated": bool(terminated),
            }
        )
        step += 1
        if terminated:
            break
        obs = obs_next

    return history, total_reward


def build_animation(histories, policy_names, title_prefix, output_path, unit_nr, rul_cap):
    fig, axes = plt.subplots(1, len(histories), figsize=(4 * max(2, len(histories)), 4), constrained_layout=True)
    if len(histories) == 1:
        axes = [axes]

    fig.suptitle(f"{title_prefix} — moteur {unit_nr} | RUL max={rul_cap}", fontsize=14)

    for ax, (name, history) in zip(axes, zip(policy_names, histories)):
        ax.set_title(name)
        ax.set_xlim(0, max(10, max((h["cycle"] for h in history), default=0) + 10))
        ax.set_ylim(0, max(20, rul_cap * 1.1))
        ax.set_xlabel("Cycle")
        ax.set_ylabel("RUL")
        ax.grid(True, alpha=0.3)

        ax.plot([], [], color="tab:blue", lw=2, label="RUL réel")
        ax.plot([], [], color="tab:orange", lw=2, label="RUL estimé")
        ax.scatter([], [], color="tab:red", s=40, label="état courant")
        ax.legend(loc="upper right")

    line_true = [ax.lines[0] for ax in axes]
    line_pred = [ax.lines[1] for ax in axes]
    marker = [ax.collections[0] for ax in axes]

    total_frames = max(len(h) for h in histories)
    frames = []
    canvas = FigureCanvasAgg(fig)

    for frame in range(total_frames):
        for ax, true_line, pred_line, marker_pt, history in zip(axes, line_true, line_pred, marker, histories):
            current_history = history[: frame + 1]
            if not current_history:
                continue
            cycles = np.array([h["cycle"] for h in current_history], dtype=float)
            true_ruls = np.array([h["true_rul"] for h in current_history], dtype=float)
            pred_ruls = np.array([h["rul_pred"] for h in current_history], dtype=float)

            true_line.set_data(cycles, true_ruls)
            pred_line.set_data(cycles, pred_ruls)
            marker_pt.set_offsets(np.array([[cycles[-1], true_ruls[-1]]]))

            last = current_history[-1]
            action_name = ACTION_LABELS.get(last["action"], str(last["action"]))
            ax.set_title(f"{policy_names[histories.index(history)]} | action={action_name} | reward={last['reward']:.1f}")

            max_cycle = max(10, int(cycles[-1]) + 10)
            ax.set_xlim(0, max_cycle)
            ax.set_ylim(0, max(20, rul_cap * 1.1))

        canvas.draw()
        arr = np.asarray(canvas.buffer_rgba())
        frames.append(arr[:, :, :3])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(str(output_path), fps=4, codec="libx264") as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"Vidéo générée : {output_path}")
    except Exception as exc:
        gif_path = output_path.with_suffix(".gif")
        imageio.imwrite(str(gif_path), np.asarray(frames), fps=4)
        print(f"MP4 export failed ({exc}); GIF enregistré : {gif_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Générer une vidéo de comparaison de politiques de maintenance.")
    parser.add_argument("--algo", choices=["rule", "dqn", "ppo", "all"], default="ppo",
                        help="Algorithme à comparer au baseline ; 'all' compare règles + dqn + ppo sur le même moteur.")
    parser.add_argument("--unit-nr", type=int, default=1, help="Numéro du moteur à simuler.")
    parser.add_argument("--output", type=str, default="videos/policy_comparison.mp4",
                        help="Chemin du fichier vidéo de sortie.")
    args = parser.parse_args()

    output_path = ROOT / args.output

    env = make_env(seed=0)
    env_cfg = load_config(str(ROOT / "configs" / "env.yaml"))
    rul_cap = env_cfg["rul_cap"]

    if args.algo == "all":
        policies = ["rule", "dqn", "ppo"]
    else:
        policies = ["rule", args.algo]

    histories = []
    titles = []

    for policy_name in policies:
        if policy_name == "rule":
            agent = load_rule_agent()
            title = "Avant (règles)"
        else:
            agent = load_rl_agent(policy_name)
            title = f"Après ({policy_name.upper()})"
        history, total_reward = run_episode(env, agent, unit_nr=args.unit_nr, max_steps=60)
        histories.append(history)
        titles.append(title)
        print(f"{title} | récompense totale={total_reward:.2f} | durée={len(history)} étapes")

    build_animation(histories, titles, "Comparaison politique de maintenance", output_path, args.unit_nr, rul_cap)


if __name__ == "__main__":
    main()
