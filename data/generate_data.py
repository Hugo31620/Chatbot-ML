"""
generate_data.py
----------------
Génère les deux fichiers du projet :
    - Raw-Data.csv     : étiquettes brutes (les valeurs possibles de chaque variable)
    - Cleaned-Data.csv : toutes les combinaisons possibles, prêtes pour l'analyse

Les 7 variables principales suivent les spécifications du sujet (directives OMS
et ministère indien de la Santé) :
    1. Country        (pays visité)
    2. Age            (tranche d'âge - groupes OMS)
    3. Symptômes OMS  : Fever, Tiredness, Dry-Cough, Difficulty-in-Breathing, Sore-Throat
    4. Autres sympt.  : Pains, Nasal-Congestion, Runny-Nose, Diarrhea, None_Sympton, None_Experiencing
    5. Severity       (None / Mild / Moderate / Severe)
    6. Contact        (Yes / No / Dont-Know)
    7. (Gender)       (Male / Female / Transgender)

La cible binaire (Covid-Positive) est construite à partir d'un score pondéré
des facteurs de risque OMS, puis bruité légèrement pour rester réaliste.
"""

import itertools
import os

import numpy as np
import pandas as pd

RNG_SEED = 42
np.random.seed(RNG_SEED)

# --------------------------------------------------------------------------- #
# 1. Définition des modalités (= contenu de Raw-Data.csv)
# --------------------------------------------------------------------------- #

AGE_GROUPS = ["0-9", "10-19", "20-24", "25-59", "60+"]

GENDERS = ["Male", "Female", "Transgender"]

# 5 symptômes OMS principaux (binaires : 1 = présent, 0 = absent)
WHO_SYMPTOMS = ["Fever", "Tiredness", "Dry-Cough", "Difficulty-in-Breathing", "Sore-Throat"]

# Autres symptômes (binaires)
OTHER_SYMPTOMS = ["Pains", "Nasal-Congestion", "Runny-Nose", "Diarrhea"]
# "None_Experiencing" = ne ressent aucun autre symptôme

SEVERITY = ["None", "Mild", "Moderate", "Severe"]

CONTACT = ["Yes", "No", "Dont-Know"]

# Liste de pays (les 10 premiers les plus impactés lors des premières vagues)
COUNTRIES = [
    "France", "Italy", "China", "USA", "Spain",
    "Germany", "UK", "India", "Iran", "Other-Country",
]


# --------------------------------------------------------------------------- #
# 2. Génération du Raw-Data : une ligne = une variable, ses modalités
# --------------------------------------------------------------------------- #

def build_raw_data() -> pd.DataFrame:
    """Reproduit la structure 'Raw-Data.csv' du dataset Kaggle d'origine :
    chaque variable y est listée avec ses modalités possibles."""
    rows = []
    for v in WHO_SYMPTOMS:
        rows.append({"Variable": v, "Type": "WHO-Symptom", "Modalities": "0 | 1"})
    for v in OTHER_SYMPTOMS:
        rows.append({"Variable": v, "Type": "Other-Symptom", "Modalities": "0 | 1"})
    rows.append({"Variable": "None_Experiencing", "Type": "Other-Symptom",
                 "Modalities": "0 | 1"})
    rows.append({"Variable": "Age", "Type": "Demographic",
                 "Modalities": " | ".join(AGE_GROUPS)})
    rows.append({"Variable": "Gender", "Type": "Demographic",
                 "Modalities": " | ".join(GENDERS)})
    rows.append({"Variable": "Severity", "Type": "Clinical",
                 "Modalities": " | ".join(SEVERITY)})
    rows.append({"Variable": "Contact", "Type": "Clinical",
                 "Modalities": " | ".join(CONTACT)})
    rows.append({"Variable": "Country", "Type": "Geographic",
                 "Modalities": " | ".join(COUNTRIES)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Génération du Cleaned-Data : toutes les combinaisons
# --------------------------------------------------------------------------- #

def build_cleaned_data() -> pd.DataFrame:
    """Crée le produit cartésien partiel des modalités.

    Pour rester dans l'ordre de grandeur du dataset Kaggle (~316k lignes),
    on combine :
        - tous les profils de symptômes OMS (2^5 = 32)
        - tous les profils d'autres symptômes (2^5 = 32, le 5e étant
          'None_Experiencing' actif uniquement si tous les autres sont à 0)
        - les 5 tranches d'âge
        - les 3 genres
        - les 4 niveaux de sévérité
        - les 3 valeurs de contact
        - 10 pays  →  on échantillonne 1 pays par ligne (pondéré)
    """
    rows = []

    who_combos = list(itertools.product([0, 1], repeat=len(WHO_SYMPTOMS)))
    other_combos = list(itertools.product([0, 1], repeat=len(OTHER_SYMPTOMS)))

    for who in who_combos:
        for other in other_combos:
            for age in AGE_GROUPS:
                for gender in GENDERS:
                    for sev in SEVERITY:
                        for ct in CONTACT:
                            # Le pays est tiré aléatoirement (l'ordre de grandeur
                            # reste proche du dataset Kaggle)
                            country = np.random.choice(COUNTRIES)

                            none_exp = 1 if sum(other) == 0 else 0

                            row = {}
                            for s, v in zip(WHO_SYMPTOMS, who):
                                row[s] = v
                            for s, v in zip(OTHER_SYMPTOMS, other):
                                row[s] = v
                            row["None_Experiencing"] = none_exp

                            row["Age"] = age
                            row["Gender"] = gender
                            row["Severity"] = sev
                            row["Contact"] = ct
                            row["Country"] = country
                            rows.append(row)

    df = pd.DataFrame(rows)
    df = _label_target(df)
    return df


# --------------------------------------------------------------------------- #
# 4. Labellisation de la cible Covid-Positive
# --------------------------------------------------------------------------- #

def _label_target(df: pd.DataFrame) -> pd.DataFrame:
    """Construit une cible binaire 'Covid-Positive' à partir d'un score de
    risque inspiré des directives OMS :

        +2 par symptôme OMS présent
        +1 par autre symptôme
        +3 si contact = Yes  /  +1 si Dont-Know
        +1 si Severity = Moderate  /  +3 si Severe
        +1 si Age = 60+    (population à risque)
        +1 si Country dans la liste des foyers majeurs (France, Italy, China,
            USA, Spain, Iran)

    On binarise au-dessus du seuil médian + bruit gaussien léger.
    """
    score = np.zeros(len(df), dtype=float)
    for s in WHO_SYMPTOMS:
        score += 2.0 * df[s].values
    for s in OTHER_SYMPTOMS:
        score += 1.0 * df[s].values

    score += np.where(df["Contact"].values == "Yes", 3.0, 0.0)
    score += np.where(df["Contact"].values == "Dont-Know", 1.0, 0.0)

    score += np.where(df["Severity"].values == "Severe", 3.0, 0.0)
    score += np.where(df["Severity"].values == "Moderate", 1.0, 0.0)
    score += np.where(df["Severity"].values == "Mild", 0.5, 0.0)

    score += np.where(df["Age"].values == "60+", 1.0, 0.0)

    hotspots = {"France", "Italy", "China", "USA", "Spain", "Iran"}
    score += np.array([1.0 if c in hotspots else 0.0 for c in df["Country"].values])

    # bruit gaussien pour casser le déterminisme parfait
    score += np.random.normal(0, 1.0, size=len(df))

    threshold = np.median(score)
    df = df.copy()
    df["Covid-Positive"] = (score > threshold).astype(int)
    return df


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    raw = build_raw_data()
    raw_path = os.path.join(out_dir, "Raw-Data.csv")
    raw.to_csv(raw_path, index=False)
    print(f"[OK] Raw-Data.csv         -> {len(raw):>7,} lignes")

    cleaned = build_cleaned_data()
    cleaned_path = os.path.join(out_dir, "Cleaned-Data.csv")
    cleaned.to_csv(cleaned_path, index=False)
    print(f"[OK] Cleaned-Data.csv     -> {len(cleaned):>7,} lignes")
    print(f"[i ] Répartition cible    -> {cleaned['Covid-Positive'].value_counts().to_dict()}")
    print(f"[i ] Colonnes ({len(cleaned.columns)}): {list(cleaned.columns)}")


if __name__ == "__main__":
    main()
