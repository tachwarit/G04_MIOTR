"""
src/plot_curves.py
Courbes d'apprentissage (moyenne ± écart-type ombré sur les seeds) à partir
de results/training_curves.csv (schéma : algo, seed, step, episode_return).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "results/training_curves.csv"
FIGURES_DIR = "figures"
N_POINTS = 200  # résolution de la grille de pas communs pour l'interpolation


def interpolate_seed(seed_df, grid):
    """Interpole la courbe (step -> episode_return) d'un seed sur une grille commune,
    nécessaire car chaque seed termine ses épisodes à des pas différents."""
    seed_df = seed_df.sort_values("step")
    return np.interp(grid, seed_df["step"], seed_df["episode_return"])


def main():
    df = pd.read_csv(CSV_PATH)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plt.figure(figsize=(7, 5))
    for algo_name, algo_df in df.groupby("algo"):
        max_step = algo_df["step"].max()
        grid = np.linspace(0, max_step, N_POINTS)

        curves = np.array([
            interpolate_seed(seed_df, grid)
            for _, seed_df in algo_df.groupby("seed")
        ])
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)

        plt.plot(grid, mean_curve, label=f"{algo_name.upper()} (n={len(curves)} seeds)")
        plt.fill_between(grid, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    plt.xlabel("Pas d'entraînement (step)")
    plt.ylabel("Retour par épisode (episode_return)")
    plt.title("Courbes d'apprentissage - moyenne ± écart-type sur les seeds")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "learning_curves.pdf")
    plt.savefig(out_path)
    print(f"Figure sauvegardée dans {out_path}")


if __name__ == "__main__":
    main()
