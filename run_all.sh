#!/usr/bin/env bash
# run_all.sh - Régénère toutes les figures, résultats et logs du projet.
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

echo "== 5. Entraînement de l'agent RL (DQN/PPO, voir configs/train.yaml) =="
python src/train.py

echo "== 6. Comparaison RL vs baseline à règles =="
python src/eval.py

echo "== 7. Test de robustesse (profils de dégradation) =="
python src/robustness_test.py

echo ""
echo "Terminé. Résultats dans results/, figures dans figures/, logs dans logs/."
