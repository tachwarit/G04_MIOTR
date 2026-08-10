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

echo ""
echo "Terminé. Résultats dans results/, figures dans figures/."
