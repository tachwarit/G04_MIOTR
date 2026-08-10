"""
src/data.py
Téléchargement, chargement et RUL du dataset NASA C-MAPSS (sous-ensemble FD001).
Source officielle : https://data.nasa.gov/docs/legacy/CMAPSSData.zip
"""
import os
import zipfile
import urllib.request
import pandas as pd

DATA_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "CMAPSSData.zip")

INDEX_NAMES = ["unit_nr", "time_cycles"]
SETTING_NAMES = ["setting_1", "setting_2", "setting_3"]
SENSOR_NAMES = [f"s_{i}" for i in range(1, 22)]
COL_NAMES = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES


def download_data():
    """Télécharge et extrait le zip NASA si ce n'est pas déjà fait."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        print(f"Téléchargement depuis {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    marker = os.path.join(DATA_DIR, "train_FD001.txt")
    if not os.path.exists(marker):
        print("Extraction ...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DATA_DIR)


def load_fd001():
    """Charge train/test/RUL pour FD001.
    sep=r'\\s+' (regex) évite les colonnes NaN parasites que produit
    un simple sep=' ' à cause des espaces multiples en fin de ligne.
    """
    download_data()
    train = pd.read_csv(os.path.join(DATA_DIR, "train_FD001.txt"),
                         sep=r"\s+", header=None, names=COL_NAMES)
    test = pd.read_csv(os.path.join(DATA_DIR, "test_FD001.txt"),
                        sep=r"\s+", header=None, names=COL_NAMES)
    rul_test = pd.read_csv(os.path.join(DATA_DIR, "RUL_FD001.txt"),
                            sep=r"\s+", header=None, names=["RUL"])
    return train, test, rul_test


def compute_train_rul(train):
    """Ajoute la colonne RUL au train set : RUL = cycle_max_moteur - cycle_courant."""
    max_cycle = train.groupby("unit_nr")["time_cycles"].max().rename("max_cycle")
    train = train.merge(max_cycle, on="unit_nr")
    train["RUL"] = train["max_cycle"] - train["time_cycles"]
    train.drop(columns=["max_cycle"], inplace=True)
    return train


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, test, rul_test = load_fd001()
    train = compute_train_rul(train)

    print("train    :", train.shape)
    print("test     :", test.shape)
    print("rul_test :", rul_test.shape)
    print(train.head())

    os.makedirs("figures", exist_ok=True)

    # Distribution des durées de vie totales
    life = train.groupby("unit_nr")["time_cycles"].max()
    plt.figure(figsize=(6, 4))
    life.hist(bins=20)
    plt.xlabel("Durée de vie (cycles)")
    plt.ylabel("Nombre de moteurs")
    plt.title("Distribution des durées de vie - FD001 train")
    plt.tight_layout()
    plt.savefig("figures/fd001_life_distribution.pdf")

    # Trajectoire de dégradation de quelques capteurs pour le moteur 1
    # (un subplot par capteur : les échelles brutes sont trop différentes
    # pour être lisibles sur un seul graphique - ex: s_3 ~1500 vs s_15 ~8)
    unit1 = train[train.unit_nr == 1]
    sensors_to_plot = ["s_2", "s_3", "s_4", "s_7", "s_11", "s_15"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for ax, s in zip(axes.flat, sensors_to_plot):
        ax.plot(unit1.time_cycles, unit1[s])
        ax.set_title(s)
        ax.set_xlabel("Cycle")
    fig.suptitle("Trajectoire de dégradation - moteur 1 (FD001)")
    plt.tight_layout()
    plt.savefig("figures/fd001_unit1_sensors.pdf")

    print("Figures sauvegardées dans figures/")
