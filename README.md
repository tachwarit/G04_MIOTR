# G04_MIOTR — Predictive Maintenance Agent

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
│   ├── train.py                  # entraînement de l'agent RL (à venir)
│   └── eval.py                    # évaluation / comparaison (à venir)
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
entraînement du modèle RUL, test de l'environnement Gymnasium, et
évaluation de l'agent baseline à règles. Il régénère tous les fichiers de
`results/` et `figures/`.

## Exécution sur Google Colab

```bash
!git clone https://github.com/tachwarit/G04_MIOTR.git
%cd G04_MIOTR
!pip install -r requirements.txt
!bash run_all.sh
```

## État d'avancement

**Milestone 1 — Scaffolding et pipeline (terminé)**
- [x] Pipeline de données NASA C-MAPSS FD001
- [x] Modèle RUL Random Forest (RMSE ≈ 18.2 cycles, MAE ≈ 13.3 cycles)
- [x] Environnement Gymnasium custom (état/actions/récompense)
- [x] Agent baseline à règles (seuils sur le RUL estimé)

**Milestone 2 — Agent RL appris (à venir)**
- [ ] Entraînement DQN/PPO (`src/train.py`)
- [ ] Comparaison quantitative agent RL vs baseline à règles (`src/eval.py`)
- [ ] Vidéos avant/après entraînement

## Configuration

Tous les hyperparamètres et seuils sont dans `configs/*.yaml` — aucune
valeur n'est codée en dur dans le code source.

## Source des données

NASA Prognostics Center of Excellence, jeu de données C-MAPSS FD001
(domaine public) : https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
