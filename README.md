# Predictive Maintenance Agent for Autonomous Condition-Based Servicing of Industrial Robotic Fleets

Agent de maintenance prédictive pour flottes de robots industriels, par
apprentissage par renforcement, appliqué au dataset NASA C-MAPSS (FD001).

**Groupe** : G04 — Master IoTR, Faculté Polydisciplinaire de Béni Mellal
**Année universitaire** : 2025-2026

## Contexte

Ce projet compare une politique de maintenance à règles fixes (baseline) à
une politique apprise par renforcement (DQN/PPO), à partir d'une estimation
du RUL (Remaining Useful Life) prédite par un modèle Random Forest. Voir
`report.pdf` pour le détail méthodologique complet.

## Structure du dépôt

```
G04_MIOTR/
├── README.md            # ce fichier
├── requirements.txt      # dépendances Python (versions figées)
├── run_all.sh             # régénère tous les résultats en une commande
├── configs/                # hyperparamètres, seuils (YAML)
├── src/
│   ├── config.py            # chargement des configs YAML
│   ├── data.py               # téléchargement + chargement FD001
│   ├── rul_model.py           # modèle RUL (Random Forest)
│   ├── env.py                  # environnement Gymnasium custom
│   ├── rule_agent.py            # agent baseline à règles
│   ├── train.py                  # entraînement DQN/PPO multi-seed
│   ├── eval.py                    # comparaison RL vs baseline à règles
│   ├── plot_curves.py              # courbes d'apprentissage (mean ± std)
│   └── robustness_test.py           # robustesse par profil de dégradation
├── results/                # métriques et prédictions (.csv)
├── logs/                    # logs TensorBoard / W&B
├── figures/                  # figures vectorielles (.pdf)
├── videos/                    # agent avant/après entraînement (.mp4)
├── notebook.md                 # journal de bord hebdomadaire
└── report.pdf                    # rapport final
```

## Installation

```bash
git clone https://github.com/tachwarit/G04_MIOTR.git
cd G04_MIOTR
pip install -r requirements.txt
```

Le dataset NASA C-MAPSS n'est pas versionné (voir `.gitignore`) : il est
téléchargé automatiquement au premier lancement depuis la source officielle
(`https://data.nasa.gov/docs/legacy/CMAPSSData.zip`).

## Reproduire tous les résultats en une commande

```bash
bash run_all.sh
```

Ce script exécute dans l'ordre : téléchargement/exploration des données,
entraînement du modèle RUL, test de l'environnement Gymnasium, évaluation de
l'agent baseline à règles, entraînement DQN et PPO (3 seeds chacun),
génération des courbes d'apprentissage (moyenne ± écart-type), comparaison
RL vs baseline, et test de robustesse par profil de dégradation. Il régénère
tous les fichiers de `results/`, `figures/` et `logs/`.

⚠️ L'entraînement multi-seed (2 algos × 3 seeds) prend environ 1 heure sur
CPU. Réduire `total_timesteps` dans `configs/train.yaml` pour un test rapide.

## Exécution sur Google Colab

```bash
!git clone https://github.com/tachwarit/G04_MIOTR.git
%cd G04_MIOTR
!pip install -r requirements.txt
!bash run_all.sh
```

## Générer une vidéo de démonstration

```bash
python src/generate_video.py --algo ppo --output videos/ppo_vs_rule.mp4
# ou, pour comparer les trois politiques sur le même moteur :
python src/generate_video.py --algo all --output videos/comparison.mp4
```

Le script charge automatiquement le dataset FD001, reconstruit le modèle RUL,
charge le modèle RL sauvegardé dans `results/` (ou le baseline à règles), puis
produit une vidéo MP4 dans `videos/` pour visualiser les décisions d'entretien
avant/après entraînement.

## État d'avancement

**Milestone 1 — Scaffolding et pipeline (terminé)**
- [x] Pipeline de données NASA C-MAPSS FD001
- [x] Modèle RUL Random Forest (RMSE ≈ 18.2 cycles, MAE ≈ 13.3 cycles)
- [x] Environnement Gymnasium custom (état/actions/récompense)
- [x] Agent baseline à règles (seuils sur le RUL estimé)

**Milestone 2 — Agent RL appris (quasi terminé)**
- [x] Entraînement DQN et PPO, 3 seeds chacun (`src/train.py`)
- [x] Courbes d'apprentissage moyenne ± écart-type (`src/plot_curves.py`)
- [x] Comparaison quantitative agent RL vs baseline à règles (`src/eval.py`) —
      DQN 10.96 / PPO 10.47 / règles 6.32 (récompense moyenne, 100 épisodes,
      0 panne pour les trois)
- [x] Test de robustesse par profil de dégradation (`src/robustness_test.py`)
- [ ] Vidéos avant/après entraînement
- [ ] `report.pdf`

## Configuration

Tous les hyperparamètres et seuils sont dans `configs/*.yaml` — aucune
valeur n'est codée en dur dans le code source.

## Source des données

NASA Prognostics Center of Excellence, jeu de données C-MAPSS FD001
(domaine public) : https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
