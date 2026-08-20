# Notebook — Journal de bord

Format par entrée : **Date** (à récupérer via `git log`, voir note en bas) /
**Claim** (hypothèse de départ) / **Evidence** (résultat obtenu) /
**Decision** (action prise en conséquence).

---

## Entrée 1 — Mise en place du projet

**Date** : 09/08/2026
**Claim** : le projet a besoin d'une structure de dépôt propre et reproductible
dès le départ, pour respecter les critères de rendu.
**Evidence** : arborescence `G04_MIOTR/` créée (configs/, src/, results/, logs/,
figures/, videos/), dépôt Git initialisé, connecté à GitHub, premier push réussi.
**Decision** : construire le pipeline module par module dans `src/`, en
committant à chaque étape validée plutôt qu'en une seule fois à la fin.

---

## Entrée 2 — Pipeline de données NASA C-MAPSS FD001

**Date** : 10/08/2026
**Claim** : le fichier officiel NASA (`CMAPSSData.zip`) est directement
téléchargeable et exploitable avec `pandas`.
**Evidence** : train (20631, 27), test (13096, 26), rul_test (100, 1) ; durées
de vie des moteurs comprises entre 150 et 350 cycles. Premier essai de
visualisation des capteurs illisible (échelles brutes trop différentes,
ex. s_3 ≈ 1600 vs s_15 ≈ 8) — corrigé avec un subplot par capteur.
**Decision** : dataset validé. Passage à la modélisation du RUL.

---

## Entrée 3 — Modèle RUL (Random Forest)

**Date** : 10/08/2026
**Claim** : un Random Forest, avec RUL plafonné à 125 cycles à l'entraînement
et évaluation sur le dernier cycle observé de chaque moteur test, donnera une
baseline raisonnable.
**Evidence** : RMSE = 18.22 cycles, MAE = 13.34 cycles sur 100 moteurs test —
cohérent avec les baselines publiées sur FD001 (15-20 de RMSE). Le nuage de
points prédit-vs-réel montre un plafonnement attendu pour les moteurs très
sains (conséquence du RUL_CAP).
**Decision** : modèle accepté comme baseline. Utilisé pour générer le RUL
estimé à chaque cycle, en entrée de l'environnement RL.

---

## Entrée 4 — Environnement Gymnasium + agent à règles

**Date** : 10/08/2026
**Claim** : un état à 3 dimensions (RUL estimé, cycle normalisé, tendance) et
3 actions (continuer / maintenance / arrêt) suffisent pour représenter le
problème, et un agent à seuils fixes (30 / 5 cycles) donne une baseline
fonctionnelle.
**Evidence** : agent à règles — 0 panne, récompense moyenne 180.8 (écart-type
40.4) sur 100 épisodes.
**Decision** : Milestone 1 (scaffolding) jugé complet. Externalisation de tous
les hyperparamètres vers `configs/*.yaml`, rédaction du README et de
`run_all.sh`.

---

## Entrée 5 — Premier entraînement DQN : détection d'un reward hacking

**Date** : 10/08/2026
**Claim** : un DQN entraîné sur l'environnement devrait apprendre à planifier
la maintenance au bon moment, au moins aussi bien que la baseline à règles.
**Evidence** : évaluation déterministe sur 100 épisodes → **100% de pannes non
planifiées**, mais récompense moyenne 154.0 (proche des 181.3 de la
baseline). Diagnostic : `continue_reward=1.0` cumulé sur ~200 cycles
dépassait largement la pénalité de panne unique (-50), incitant l'agent à
toujours continuer.
**Decision** : correction du reward shaping — `continue_reward` passé de 1.0
à 0.0 dans `configs/env.yaml` (simple changement de config, aucun code
modifié). Ré-entraînement.

---

## Entrée 6 — Validation de la correction + comparaison RL vs baseline

**Date** :  11/08/2026
**Claim** : avec `continue_reward=0`, DQN et PPO devraient apprendre une
politique de maintenance fiable et efficace.
**Evidence** : DQN (10.96) et PPO (10.47) battent tous les deux la baseline
(6.32), avec 0 panne dans les trois cas. Le RL attend en moyenne un RUL réel
plus bas avant maintenance (~13 cycles) que le seuil fixe (~29 cycles) —
meilleure exploitation de la durée de vie.
**Decision** : les deux agents RL sont retenus comme résultat principal, DQN
légèrement devant PPO. Passage au test de robustesse.

---

## Entrée 7 — Test de robustesse (profils de dégradation)

**Date** : 11/08/2026
**Claim** : les politiques (règles, DQN, PPO) doivent rester stables sur des
moteurs à dégradation rapide, normale et lente (tertiles de durée de vie).
**Evidence** : 9 combinaisons testées (3 profils × 3 politiques, 100 épisodes
chacune) — 0 panne partout, avantage du RL sur la baseline maintenu dans les
3 profils. Légère baisse de récompense pour les 3 politiques sur les moteurs
à vie longue, probablement liée à une précision RUL plus faible sur cet
horizon (limite du modèle, pas de la politique).
**Decision** : critère de robustesse validé. Limite du modèle RUL sur les
horizons longs à documenter dans le rapport.

---

## Entrée 8 — Nettoyage et synchronisation du dépôt

**Date** : 11/08/2026
**Claim** : tous les résultats générés localement doivent être versionnés sur
GitHub pour la remise finale.
**Evidence** : `git status` a révélé plusieurs fichiers (figures, logs,
résultats) non commités, plus des fichiers `__pycache__/` à exclure.
**Decision** : ajout de `__pycache__/` et `*.pyc` au `.gitignore`,
synchronisation complète du dépôt.

---
---

## Entrée 9 — Adaptation du gabarit de rapport M122

**Date** : 12/08/2026
**Claim** : le gabarit de rapport du module (reproduction d'une baseline
publiée + amélioration proposée) doit pouvoir s'adapter à un projet basé
sur un environnement original, sans baseline externe publiée.
**Evidence** : rédaction section par section (LaTeX puis converti en Word)
en mappant explicitement baseline = agent à règles (interne, définie dans
le cahier des charges) et enhancement = agent RL (DQN/PPO) — chaque
divergence par rapport au gabarit documentée explicitement dans le texte
(Sections II-C, IV).
**Decision** : rapport structuré en 9 sections + annexes selon ce mapping,
avec écarts assumés et justifiés plutôt que gabarit rempli artificiellement.

---

## Entrée 10 — Découverte d'une instabilité inter-seeds critique

**Date** : 18/08/2026
**Claim** : en corrigeant `train.py` pour sauvegarder un modèle par seed
(au lieu du seul seed 0), le nouveau seed 0 devrait reproduire les
résultats déjà obtenus (DQN ≈ 10.96, PPO ≈ 10.47, 0 panne).
**Evidence** : le nouveau seed 0 donne des résultats très différents
(DQN=1.82, PPO=13.03), et les seeds 1/2 de PPO montrent 36 et 40 pannes
sur 100 — la moyenne agrégée sur 3 seeds (DQN 0.63±1.22, PPO −2.29±15.07)
ne dépasse plus la baseline (6.25±0.23). Phénomène cohérent avec la
littérature sur la reproductibilité en RL profond (Henderson et al. 2018).
**Decision** : réécriture honnête de l'Abstract, du Tableau 4, de la
Discussion et de la Conclusion pour refléter cette instabilité plutôt que
de garder les anciens chiffres non reproductibles.

---

## Entrée 11 — Ablation à 3 niveaux de `continue_reward`

**Date** : 19/08/2026
**Claim** : une valeur intermédiaire de `continue_reward` (0.5) devrait
donner un comportement intermédiaire entre 0.0 (sûr) et 1.0 (catastrophe).
**Evidence** : à 0.5, 0 panne sur 100 épisodes, mais récompense moyenne de
106.8 — bien au-delà de ce qu'une maintenance bien timée peut expliquer
(max ~15 pts). Hypothèse : l'agent repousse la maintenance très près de la
limite de panne sans jamais la franchir sur ces épisodes précis.
**Decision** : Tableau 5 et Fig. 4 (3 points) documentant une transition
non-linéaire, pas progressive, entre 0.0 et 1.0.

---

## Entrée 12 — Intégration de `rliable` et conflit de dépendances

**Date** : 19/08/2026
**Claim** : l'IQM et l'IC bootstrap stratifié demandés par le protocole du
module peuvent être calculés directement avec la librairie `rliable`.
**Evidence** : `rliable` plante avec `pandas==3.0.5` (conflit avec sa
dépendance `arch`) ; fonctionne avec `pandas<2.2`. Une fois résolu, les IC
bootstrap (bornés par le min/max observé à n=3 seeds) sont proches des
approximations normales utilisées provisoirement, mais plus rigoureux.
**Decision** : script séparé (`compute_rliable_stats.py`), volontairement
exclu de `run_all.sh`, avec la procédure de contournement documentée dans
son en-tête.

---

## Entrée 13 — Vidéo avant/après entraînement

**Date** : 19/08/2026
**Claim** : une politique aléatoire ("avant entraînement") tournera
suffisamment longtemps pour être comparée visuellement à l'agent DQN
entraîné sur la même trajectoire de moteur.
**Evidence** : la politique aléatoire a ~2/3 de chances de choisir une
action terminale (maintenance/arrêt) dès le premier cycle — la plupart des
tirages s'arrêtent en 1 cycle, ce qui ne donne rien à animer.
**Decision** : `make_comparison_video.py` réessaie plusieurs tirages
aléatoires jusqu'à en trouver un assez long à montrer (politique toujours
100% aléatoire, seul l'exemple retenu est choisi pour être lisible).

---
## Récupérer les vraies dates

Chaque entrée correspond à un commit précis. Pour retrouver la date réelle de
chacune, lance dans `G04_MIOTR/` :

```bash
git log --format="%ad  %s" --date=short
```

Fais correspondre le message de commit à l'entrée ci-dessus (ex. le commit
`"Add Random Forest RUL model (RMSE=18.2, MAE=13.3)"` donne la date de
l'Entrée 3), puis remplace les `JJ/MM/2026`.
