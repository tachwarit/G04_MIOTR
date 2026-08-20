"""
src/make_performance_profile.py
Génère figures/performance_profile.pdf (Fig. 3 du rapport) : pour chaque
politique, la fraction des seeds dont le retour moyen dépasse un seuil τ,
balayé sur une plage continue. Lit les données directement depuis
results/baseline_seed_sweep.csv et results/seed_sweep_eval.csv (déjà
générés par eval_baseline_seeds.py et eval_seed_sweep.py) plutôt que des
valeurs codées en dur.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"rule_based": "#4C72B0", "dqn": "#55A868", "ppo": "#C44E52"}


def load_scores():
    baseline = pd.read_csv("results/baseline_seed_sweep.csv")
    baseline = baseline[baseline.seed != "mean"]
    scores = {"rule_based": baseline["mean_reward"].to_numpy()}

    sweep = pd.read_csv("results/seed_sweep_eval.csv")
    for algo in ["dqn", "ppo"]:
        scores[algo] = sweep[sweep.algo == algo]["mean_reward"].to_numpy()
    return scores


def main():
    scores = load_scores()
    all_vals = np.concatenate(list(scores.values()))
    tau = np.linspace(all_vals.min() - 1, all_vals.max() + 1, 400)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, vals in scores.items():
        frac = np.array([(vals >= t).mean() for t in tau])
        ax.step(tau, frac, where="post", label=f"{name} (n={len(vals)} seeds)",
                 color=COLORS.get(name, "gray"), linewidth=2)

    ax.set_xlabel("Seuil de performance τ (récompense moyenne)")
    ax.set_ylabel("Fraction des runs (seeds) ≥ τ")
    ax.set_title("Performance profile (seeds officiels)")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out_path = "figures/performance_profile.pdf"
    plt.savefig(out_path)
    print(f"Figure sauvegardée dans {out_path}")


if __name__ == "__main__":
    main()
