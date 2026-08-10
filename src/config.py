"""
src/config.py
Chargement des fichiers de configuration YAML (configs/*.yaml).
"""
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
