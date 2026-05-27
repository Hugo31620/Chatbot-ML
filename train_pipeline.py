"""
train_pipeline.py
-----------------
Pipeline ML complet :
    1. EDA + graphiques (sauvegardés en PNG)
    2. Préprocessing (one-hot)
    3. Apprentissage supervisé (LogReg, RandomForest, GradientBoosting)
    4. Apprentissage non supervisé (K-Means + PCA)
    5. Sauvegarde des artefacts (modèles + encodeur) pour l'app Streamlit

Sortie :
    models/
        ├─ best_model.joblib
        ├─ kmeans.joblib
        ├─ encoder.joblib
        ├─ feature_names.joblib
        ├─ cluster_profiles.joblib
        └─ metrics.joblib
    assets/
        ├─ target_distribution.png
        ├─ symptoms_distribution.png
        ├─ correlation_heatmap.png
        ├─ models_comparison.png
        ├─ confusion_matrix.png
        ├─ feature_importance.png
        ├─ elbow_silhouette.png
        └─ pca_clusters.png
"""
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "Cleaned-Data.csv")
MODELS_DIR = os.path.join(ROOT, "models")
ASSETS_DIR = os.path.join(ROOT, "assets")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# 1. Chargement
# --------------------------------------------------------------------------- #
print("[1/6] Chargement du dataset...")
df = pd.read_csv(DATA_PATH)
print(f"      Lignes : {len(df):,}  /  Colonnes : {df.shape[1]}")

WHO_SYMPTOMS = ["Fever", "Tiredness", "Dry-Cough", "Difficulty-in-Breathing", "Sore-Throat"]
OTHER_SYMPTOMS = ["Pains", "Nasal-Congestion", "Runny-Nose", "Diarrhea"]
TARGET = "Covid-Positive"
CATEGORICAL = ["Age", "Gender", "Severity", "Contact", "Country"]

# --------------------------------------------------------------------------- #
# 2. EDA
# --------------------------------------------------------------------------- #
print("[2/6] EDA + visualisations...")

# 2.1 Distribution de la cible
plt.figure(figsize=(6, 4))
ax = sns.countplot(x=TARGET, data=df, palette=["#5DADE2", "#E74C3C"])
ax.set_xticklabels(["Négatif", "Positif"])
ax.set_title("Répartition de la cible Covid-Positive")
ax.set_xlabel("")
ax.set_ylabel("Nombre de profils")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "target_distribution.png"), dpi=110)
plt.close()

# 2.2 Distribution des symptômes OMS
plt.figure(figsize=(9, 4))
sym_counts = df[WHO_SYMPTOMS].sum().sort_values(ascending=True)
ax = sym_counts.plot(kind="barh", color="#E67E22")
ax.set_title("Fréquence d'apparition des 5 symptômes OMS")
ax.set_xlabel("Nombre de profils concernés")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "symptoms_distribution.png"), dpi=110)
plt.close()

# 2.3 Heatmap de corrélation (symptômes + cible)
plt.figure(figsize=(8, 6))
corr = df[WHO_SYMPTOMS + OTHER_SYMPTOMS + [TARGET]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            cbar_kws={"label": "Corrélation de Pearson"})
plt.title("Corrélations symptômes ↔ Covid-Positive")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "correlation_heatmap.png"), dpi=110)
plt.close()

# --------------------------------------------------------------------------- #
# 3. Préprocessing : one-hot
# --------------------------------------------------------------------------- #
print("[3/6] Préprocessing (one-hot encoding)...")

X = df.drop(columns=[TARGET])
y = df[TARGET].values
X_encoded = pd.get_dummies(X, columns=CATEGORICAL, drop_first=False)
feature_names = list(X_encoded.columns)
print(f"      Features finales : {len(feature_names)}")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded.values, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
)
print(f"      Train : {X_train.shape[0]:,}  /  Test : {X_test.shape[0]:,}")

# --------------------------------------------------------------------------- #
# 4. Apprentissage supervisé
# --------------------------------------------------------------------------- #
print("[4/6] Entraînement des modèles supervisés...")

models = {
    "LogisticRegression": LogisticRegression(max_iter=500, n_jobs=-1, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=120, max_depth=12,
                                           n_jobs=-1, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=120, max_depth=5,
                                                   random_state=RANDOM_STATE),
}

results = []
fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results.append({
        "Modèle": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    })
    fitted[name] = model
    print(f"      - {name:20s} acc={results[-1]['Accuracy']:.3f}  "
          f"f1={results[-1]['F1-score']:.3f}  auc={results[-1]['ROC-AUC']:.3f}")

results_df = pd.DataFrame(results).set_index("Modèle")

# 4.1 Comparaison
fig, ax = plt.subplots(figsize=(9, 5))
results_df.plot(kind="bar", ax=ax, edgecolor="white")
ax.set_title("Comparaison des modèles supervisés")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", ncol=5, fontsize=8)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "models_comparison.png"), dpi=110)
plt.close()

# 4.2 Meilleur modèle (sur F1)
best_name = results_df["F1-score"].idxmax()
best_model = fitted[best_name]
print(f"      → Meilleur modèle : {best_name}")

# Matrice de confusion
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
            xticklabels=["Négatif", "Positif"], yticklabels=["Négatif", "Positif"])
plt.title(f"Matrice de confusion - {best_name}")
plt.ylabel("Vrai")
plt.xlabel("Prédit")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "confusion_matrix.png"), dpi=110)
plt.close()

# 4.3 Feature importance (tree-based) ou coefficients (linéaires)
imp = None
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=feature_names)
    imp_title = f"Top 15 features - {best_name} (importance)"
elif hasattr(best_model, "coef_"):
    imp = pd.Series(np.abs(best_model.coef_[0]), index=feature_names)
    imp_title = f"Top 15 features - {best_name} (|coefficients|)"

if imp is not None:
    imp = imp.sort_values(ascending=True).tail(15)
    plt.figure(figsize=(8, 6))
    imp.plot(kind="barh", color="#16A085")
    plt.title(imp_title)
    plt.xlabel("Poids")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "feature_importance.png"), dpi=110)
    plt.close()

# --------------------------------------------------------------------------- #
# 5. Apprentissage non supervisé : K-Means
# --------------------------------------------------------------------------- #
print("[5/6] Clustering K-Means...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded.values)

# 5.1 Méthode du coude + silhouette
ks = list(range(2, 7))
inertias, sils = [], []
# Sous-échantillon pour la silhouette (coût élevé)
sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_scaled), size=5000, replace=False)
for k in ks:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled[sample_idx], km.labels_[sample_idx]))
    print(f"      k={k}  inertia={km.inertia_:>12,.0f}  silhouette={sils[-1]:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(ks, inertias, marker="o", color="#2980B9")
ax1.set_title("Méthode du coude")
ax1.set_xlabel("k")
ax1.set_ylabel("Inertie")
ax2.plot(ks, sils, marker="s", color="#C0392B")
ax2.set_title("Score de silhouette")
ax2.set_xlabel("k")
ax2.set_ylabel("Silhouette")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "elbow_silhouette.png"), dpi=110)
plt.close()

# 5.2 K final = celui qui maximise la silhouette
best_k = ks[int(np.argmax(sils))]
print(f"      → k retenu : {best_k}")
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
labels = kmeans.labels_

# 5.3 PCA 2D pour visualisation
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled[sample_idx])
plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels[sample_idx],
                palette="tab10", s=14, alpha=0.7, legend="full")
plt.title(f"Clusters K-Means (k={best_k}) — projection PCA 2D")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "pca_clusters.png"), dpi=110)
plt.close()

# 5.4 Profilage des clusters (numérique + catégoriel)
df_clusters = df.copy()
df_clusters["cluster"] = labels

# Profil quantitatif : moyenne des symptômes + taux de COVID par cluster
profile_num = df_clusters.groupby("cluster").agg(
    taille=(TARGET, "size"),
    taux_covid=(TARGET, "mean"),
    nb_symptomes_oms=(WHO_SYMPTOMS[0], lambda _: df_clusters.loc[_.index, WHO_SYMPTOMS].sum(axis=1).mean()),
    fievre=("Fever", "mean"),
    fatigue=("Tiredness", "mean"),
    toux_seche=("Dry-Cough", "mean"),
    diff_resp=("Difficulty-in-Breathing", "mean"),
    mal_gorge=("Sore-Throat", "mean"),
).round(3)

# Profil catégoriel : modalité dominante par cluster
def _mode(s):
    return s.value_counts().idxmax()

profile_cat = df_clusters.groupby("cluster").agg(
    age_dominant=("Age", _mode),
    severity_dominante=("Severity", _mode),
    contact_dominant=("Contact", _mode),
    pays_dominant=("Country", _mode),
)

profile = pd.concat([profile_num, profile_cat], axis=1)
print("\n--- Profilage des clusters ---")
print(profile.to_string())

# --------------------------------------------------------------------------- #
# 6. Sauvegarde
# --------------------------------------------------------------------------- #
print("[6/6] Sauvegarde des modèles...")
joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans.joblib"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.joblib"))
joblib.dump(profile, os.path.join(MODELS_DIR, "cluster_profiles.joblib"))
joblib.dump({
    "best_name": best_name,
    "results": results_df.to_dict(),
    "best_k": int(best_k),
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
}, os.path.join(MODELS_DIR, "metrics.joblib"))

print("\n[OK] Pipeline terminé. Artefacts sauvegardés.")
print(f"     Modèle retenu       : {best_name}")
print(f"     Nombre de clusters  : {best_k}")
print(f"     F1 test             : {results_df.loc[best_name, 'F1-score']:.3f}")
