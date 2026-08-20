#!/usr/bin/env bash
# run_all.sh - Régénère toutes les figures, résultats et logs du projet.
#
# NOTE : deux scripts sont volontairement EXCLUS de ce pipeline :
#   - src/compute_rliable_stats.py : nécessite pandas<2.2, incompatible avec
#     le pandas==3.0.5 pinné dans requirements.txt (conflit avec la
#     dépendance "arch" de rliable). À lancer manuellement, séparément
#     (voir l'en-tête du script pour la marche à suivre).
#   - src/ablation_continue_reward.py : exploratoire, prend une valeur en
#     argument CLI (ex: 0.5), pas destiné à tourner automatiquement.
set -e

cd "$(dirname "$0")"

echo "== 1. Pipeline de données + exploration =="
python src/data.py

echo "== 2. Modèle RUL (Random Forest) =="
python src/rul_model.py

echo "== 3. Test de l'environnement Gymnasium =="
python src/env.py

echo "== 4. Agent baseline à règles =="
python src/rule_agent.py

echo "== 5. Baseline sur les seeds officiels (résultats + Tableau 3) =="
python src/eval_baseline_seeds.py

echo "== 6. Entraînement de l'agent RL (DQN/PPO, multi-seed, voir configs/train.yaml) =="
python src/train.py

echo "== 6b. Courbes d'apprentissage (moyenne ± écart-type sur les seeds) =="
python src/plot_curves.py

echo "== 7. DQN/PPO sur les seeds officiels (résultats + Tableau 6) =="
python src/eval_seed_sweep.py

echo "== 8. Comparaison RL vs baseline à règles (Tableau 4, Fig. 2) =="
python src/eval.py

echo "== 9. Test de robustesse (profils de dégradation) =="
python src/robustness_test.py

echo "== 10. Performance profile (Fig. 3) =="
python src/make_performance_profile.py

echo "== 11. Vidéo avant/après entraînement =="
python src/make_comparison_video.py

echo ""
echo "Terminé. Résultats dans results/, figures dans figures/, vidéo dans videos/, logs dans logs/."
echo "Rappel : src/compute_rliable_stats.py et src/ablation_continue_reward.py"
echo "s'exécutent séparément (voir leurs en-têtes)."
