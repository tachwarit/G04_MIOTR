# Predictive Maintenance Agent for Autonomous Condition-Based Servicing of Industrial Robotic Fleets

Agent de maintenance prédictive pour flottes de robots industriels, par
apprentissage par renforcement, appliqué au dataset NASA C-MAPSS (FD001).

**Groupe** : G04_MIOTR — Master IoTR, Faculté Polydisciplinaire de Béni Mellal
**Année universitaire** : 2025-2026

## Contexte

Ce projet compare une politique de maintenance à règles fixes (baseline
interne, aucun papier publié n'existe sur cet environnement original) à
une politique apprise par renforcement (DQN/PPO), à partir d'une
estimation du RUL (Remaining Useful Life) prédite par un modèle Random
Forest. Voir `report.pdf` pour le détail méthodologique complet, y
compris une découverte importante d'instabilité inter-seeds documentée
honnêtement (Sections VI et VIII).

## Structure du dépôt

```
G04_MIOTR/
├── README.md                    # ce fichier
├── requirements.txt              # dépendances Python (versions figées)
├── run_all.sh                     # régénère (presque) tous les résultats en une commande
├── configs/                        # hyperparamètres, seuils (YAML)
├── src/
│   ├── config.py                     # chargement des configs YAML
│   ├── data.py                        # téléchargement + chargement FD001
│   ├── rul_model.py                    # modèle RUL (Random Forest)
│   ├── env.py                           # environnement Gymnasium custom
│   ├── rule_agent.py                     # agent baseline à règles
│   ├── eval_baseline_seeds.py             # baseline sur les 3 seeds officiels
│   ├── train.py                            # entraînement DQN/PPO multi-seed
│   ├── plot_curves.py                       # courbes d'apprentissage (Fig. 1)
│   ├── eval_seed_sweep.py                    # DQN/PPO sur les 3 seeds officiels
│   ├── eval.py                                # comparaison RL vs baseline (Tableau 4, Fig. 2)
│   ├── robustness_test.py                      # robustesse par profil de dégradation
│   ├── make_performance_profile.py              # performance profile (Fig. 3)
│   ├── ablation_continue_reward.py               # ablation reward shaping (manuel, CLI)
│   ├── compute_rliable_stats.py                   # IQM + IC bootstrap (manuel, voir note)
│   └── make_comparison_video.py                    # vidéo avant/après entraînement
├── results/                        # métriques, prédictions, sweeps (.csv)
├── logs/                            # logs TensorBoard / Monitor
├── figures/                          # figures vectorielles (.pdf)
├── videos/                            # agent avant/après entraînement (.mp4)
├── notebook.md                         # journal de bord hebdomadaire
└── report.pdf                           # rapport final (export de report.docx)
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

## Reproduire les résultats

```bash
bash run_all.sh
```

Ce script exécute 11 étapes : données, modèle RUL, test d'environnement,
agent à règles, baseline multi-seed, entraînement DQN/PPO multi-seed,
courbes d'apprentissage, évaluation DQN/PPO multi-seed, comparaison
finale, robustesse, et performance profile — puis génère la vidéo
avant/après entraînement.

⚠️ **Deux scripts sont volontairement exclus** de `run_all.sh`, et
s'exécutent séparément (voir leur en-tête pour la marche à suivre) :
- `src/compute_rliable_stats.py` — nécessite `pandas<2.2`, incompatible
  avec le `pandas==3.0.5` pinné ici (conflit avec la dépendance `arch` de
  `rliable`).
- `src/ablation_continue_reward.py` — exploratoire, prend une valeur en
  argument CLI (`python src/ablation_continue_reward.py 0.5`).

⚠️ L'entraînement multi-seed (2 algos × 3 seeds) prend environ 1 heure sur
CPU. Réduire `total_timesteps` dans `configs/train.yaml` pour un test rapide.

## État d'avancement

**Milestone 1 — Scaffolding et pipeline (terminé)**
- [x] Pipeline de données NASA C-MAPSS FD001
- [x] Modèle RUL Random Forest (RMSE ≈ 18.2 cycles, MAE ≈ 13.3 cycles)
- [x] Environnement Gymnasium custom (état/actions/récompense)
- [x] Agent baseline à règles (seuils sur le RUL estimé)

**Milestone 2 — Agent RL appris (terminé)**
- [x] Entraînement DQN et PPO, 3 seeds chacun (`src/train.py`)
- [x] Comparaison quantitative rigoureuse (IQM + IC bootstrap `rliable`) —
      baseline 6.25 (±0.23), DQN 0.63 (±1.22), PPO −2.29 (±15.07) ;
      **résultat honnête** : forte instabilité inter-seeds découverte en
      cours de route (voir `notebook.md`, entrée 10), documentée plutôt
      que masquée
- [x] Ablation du reward shaping (`continue_reward` à 0.0/0.5/1.0) —
      transition sécurité → catastrophe non-linéaire
- [x] Test de robustesse par profil de dégradation
- [x] Performance profile (Fig. 3)
- [x] Vidéo avant/après entraînement

## Configuration

Tous les hyperparamètres et seuils sont dans `configs/*.yaml` — aucune
valeur n'est codée en dur dans le code source.

## Rapport final

Le rapport est rédigé dans `report.docx` (9 sections + annexes, suivant le
gabarit M122 adapté à ce projet original). Pour la remise : `Fichier >
Enregistrer sous > PDF` dans Word, renommer en `report.pdf`, placer à la
racine du dépôt.

## Source des données

NASA Prognostics Center of Excellence, jeu de données C-MAPSS FD001
(domaine public) : https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
