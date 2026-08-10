"""
src/rul_model.py
Modèle de prédiction du RUL (Random Forest Regressor) pour NASA C-MAPSS FD001.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data import load_fd001, compute_train_rul, SENSOR_NAMES, SETTING_NAMES

# Cap standard dans la littérature C-MAPSS : au-delà, le RUL réel est peu
# prédictible à partir des capteurs (dégradation pas encore visible).
RUL_CAP = 125
RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def select_features(train, variance_threshold=1e-4):
    """Écarte les capteurs quasi constants (aucune info utile pour le RUL)."""
    variances = train[SENSOR_NAMES].var()
    kept = variances[variances > variance_threshold].index.tolist()
    dropped = [s for s in SENSOR_NAMES if s not in kept]
    print(f"Capteurs retenus ({len(kept)}): {kept}")
    print(f"Capteurs écartés, quasi constants ({len(dropped)}): {dropped}")
    return kept


def build_train_set(train, feature_cols):
    X = train[feature_cols]
    y = train["RUL"].clip(upper=RUL_CAP)
    return X, y


def build_test_set(test, rul_test, feature_cols):
    # Protocole standard C-MAPSS : un seul point par moteur, la dernière
    # ligne observée, comparé au RUL vrai fourni dans RUL_FD001.txt.
    last_cycle = test.groupby("unit_nr").tail(1).reset_index(drop=True)
    X = last_cycle[feature_cols]
    y = rul_test["RUL"].values
    return X, y


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    train, test, rul_test = load_fd001()
    train = compute_train_rul(train)

    feature_cols = SETTING_NAMES + select_features(train)

    X_train, y_train = build_train_set(train, feature_cols)
    X_test, y_test = build_test_set(test, rul_test, feature_cols)

    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"RMSE test : {rmse:.2f} cycles")
    print(f"MAE  test : {mae:.2f} cycles")

    pd.DataFrame([{
        "model": "random_forest", "rmse": rmse, "mae": mae,
        "rul_cap": RUL_CAP, "n_estimators": 200,
    }]).to_csv(os.path.join(RESULTS_DIR, "rul_metrics.csv"), index=False)

    pd.DataFrame({
        "unit_nr": test["unit_nr"].unique(),
        "rul_true": y_test, "rul_pred": y_pred,
    }).to_csv(os.path.join(RESULTS_DIR, "rul_predictions.csv"), index=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    lims = [0, max(y_test.max(), y_pred.max()) + 10]
    plt.plot(lims, lims, "r--", label="Prédiction parfaite")
    plt.xlabel("RUL réel (cycles)")
    plt.ylabel("RUL prédit (cycles)")
    plt.title(f"RUL prédit vs réel - Random Forest (RMSE={rmse:.1f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "rul_pred_vs_true.pdf"))

    print("Résultats sauvegardés dans results/ et figures/")


if __name__ == "__main__":
    main()
