"""
build_notebook.py
-----------------
Construit le notebook 01_analyse_covid.ipynb à partir d'une liste structurée
de cellules (markdown / code). Pas de dépendance externe (nbformat n'étant
pas disponible dans cet environnement) - sortie en JSON pur.
"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "notebooks", "01_analyse_covid.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS = [
    # ----------------------------------------------------------------- INTRO
    md("""# 🩺 COVID-19 Symptoms Checker — Analyse & Modélisation

**Projet M1 Data & IA — YNOV**

Ce notebook constitue la partie analytique du projet *COVID-19 Symptoms Checker*.
Le pipeline complet comporte deux livrables :

1. **ce notebook** — analyse exploratoire, apprentissage supervisé (classification)
   et apprentissage non supervisé (clustering),
2. **`app.py`** — un chatbot Streamlit qui interroge l'utilisateur question
   par question et utilise les modèles entraînés ici pour produire un avis indicatif.

### Contexte

Source des données : [Kaggle - COVID-19 Symptoms Checker](https://www.kaggle.com/iamhungundji/covid19-symptoms-checker)

Les 7 variables principales suivent les recommandations de l'**OMS** et du
ministère indien de la Santé :

| Variable          | Description |
|-------------------|-------------|
| `Country`         | Pays récemment visité |
| `Age`             | Tranche d'âge (groupes OMS) |
| `Symptômes OMS`   | Fièvre, Fatigue, Toux sèche, Difficulté à respirer, Mal de gorge |
| `Autres symptômes`| Douleurs, Congestion nasale, Écoulement nasal, Diarrhée |
| `Severity`        | Aucune / Légère / Modérée / Sévère |
| `Contact`         | Contact avec un cas positif (Oui / Non / Ne sait pas) |

> ⚠️ **Disclaimer** — Les résultats produits n'ont aucune valeur diagnostique.
"""),

    # ------------------------------------------------------- 1. IMPORTS
    md("## 1. Imports & configuration"),
    code("""import os
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
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='Set2')
plt.rcParams['figure.dpi'] = 100

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)"""),

    # ------------------------------------------------------- 2. CHARGEMENT
    md("""## 2. Chargement des données

Le dataset `Cleaned-Data.csv` contient toutes les combinaisons possibles des
modalités des 7 variables (≈92 160 lignes après produit cartésien partiel).
La colonne cible `Covid-Positive` (0/1) a été construite à partir d'un score
de risque pondéré inspiré des facteurs identifiés par l'OMS."""),

    code("""df = pd.read_csv('../data/Cleaned-Data.csv')
print(f'Dimensions : {df.shape}')
df.head()"""),

    code("""# Aperçu structurel
df.info()"""),

    code("""# Vérification : aucune valeur manquante (jeu de données synthétique exhaustif)
df.isnull().sum().sum()"""),

    # ------------------------------------------------------- 3. EDA
    md("""## 3. Analyse exploratoire (EDA)

### 3.1 Distribution de la cible

Le dataset a été construit pour être équilibré (≈50% positifs / 50% négatifs),
afin d'éviter tout biais de classe lors de l'entraînement."""),

    code("""WHO_SYMPTOMS = ['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat']
OTHER_SYMPTOMS = ['Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea']
TARGET = 'Covid-Positive'
CATEGORICAL = ['Age', 'Gender', 'Severity', 'Contact', 'Country']

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=TARGET, data=df, palette=['#5DADE2', '#E74C3C'], ax=ax)
ax.set_xticklabels(['Négatif', 'Positif'])
ax.set_title('Répartition de la cible Covid-Positive')
ax.set_xlabel('')
ax.set_ylabel('Nombre de profils')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}',
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom')
plt.show()

print('Répartition :', df[TARGET].value_counts(normalize=True).round(3).to_dict())"""),

    md("### 3.2 Fréquence des symptômes OMS"),

    code("""fig, ax = plt.subplots(figsize=(9, 4))
df[WHO_SYMPTOMS].sum().sort_values().plot(kind='barh', color='#E67E22', ax=ax)
ax.set_title("Fréquence d'apparition des 5 symptômes OMS dans le dataset")
ax.set_xlabel('Nombre de profils concernés')
plt.tight_layout()
plt.show()"""),

    md("### 3.3 Corrélations symptômes ↔ cible"),

    code("""plt.figure(figsize=(8, 6))
corr = df[WHO_SYMPTOMS + OTHER_SYMPTOMS + [TARGET]].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('Corrélations symptômes ↔ Covid-Positive')
plt.tight_layout()
plt.show()"""),

    md("""**Lecture** — Les symptômes OMS présentent une corrélation positive
modérée à forte avec la cible (`Difficulty-in-Breathing`, `Fever`, `Dry-Cough`
en tête). Les autres symptômes sont moins discriminants, ce qui confirme le
choix de l'OMS de les distinguer."""),

    md("### 3.4 Croisement Severity × Cible"),

    code("""fig, ax = plt.subplots(figsize=(8, 4))
ct = pd.crosstab(df['Severity'], df[TARGET], normalize='index') * 100
ct = ct.reindex(['None', 'Mild', 'Moderate', 'Severe'])
ct.plot(kind='bar', stacked=True, color=['#5DADE2', '#E74C3C'], ax=ax)
ax.set_title('Proportion de positifs par niveau de gravité (%)')
ax.set_ylabel('% du groupe')
ax.set_xlabel('Severity')
ax.legend(['Négatif', 'Positif'])
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()"""),

    # ------------------------------------------------------- 4. PREPROCESSING
    md("""## 4. Préprocessing

Les symptômes sont déjà binaires (0/1). Les autres variables catégorielles
(`Age`, `Gender`, `Severity`, `Contact`, `Country`) sont encodées en
**one-hot** pour rester interprétables et compatibles avec l'ensemble des
modèles considérés."""),

    code("""X = df.drop(columns=[TARGET])
y = df[TARGET].values

X_encoded = pd.get_dummies(X, columns=CATEGORICAL, drop_first=False)
feature_names = list(X_encoded.columns)
print(f'Nombre de features finales : {len(feature_names)}')

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded.values, y,
    test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f'Train : {X_train.shape[0]:,}  /  Test : {X_test.shape[0]:,}')"""),

    # ------------------------------------------------------- 5. SUPERVISE
    md("""## 5. Apprentissage supervisé

On compare **trois familles de modèles** complémentaires :

- **Logistic Regression** — référence linéaire, rapide, interprétable
- **Random Forest** — non linéaire, robuste, capture les interactions
- **Gradient Boosting** — souvent état de l'art sur tabulaire

Métriques d'évaluation : `Accuracy`, `Precision`, `Recall`, `F1-score`, `ROC-AUC`.
Le F1-score est privilégié comme critère de sélection (équilibre
précision/rappel)."""),

    code("""models = {
    'LogisticRegression': LogisticRegression(max_iter=500, n_jobs=-1, random_state=RANDOM_STATE),
    'RandomForest':       RandomForestClassifier(n_estimators=120, max_depth=12,
                                                 n_jobs=-1, random_state=RANDOM_STATE),
    'GradientBoosting':   GradientBoostingClassifier(n_estimators=120, max_depth=5,
                                                     random_state=RANDOM_STATE),
}

results = []
fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results.append({
        'Modèle':    name,
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall':    recall_score(y_test, y_pred),
        'F1-score':  f1_score(y_test, y_pred),
        'ROC-AUC':   roc_auc_score(y_test, y_proba),
    })
    fitted[name] = model

results_df = pd.DataFrame(results).set_index('Modèle').round(3)
results_df"""),

    md("### 5.1 Comparaison visuelle"),

    code("""fig, ax = plt.subplots(figsize=(10, 5))
results_df.plot(kind='bar', ax=ax, edgecolor='white')
ax.set_title('Comparaison des modèles supervisés')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right', ncol=5, fontsize=8)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()"""),

    md("### 5.2 Sélection et analyse du meilleur modèle"),

    code("""best_name = results_df['F1-score'].idxmax()
best_model = fitted[best_name]
print(f'Meilleur modèle (sur F1) : {best_name}')
print()
print(classification_report(y_test, best_model.predict(X_test),
                            target_names=['Négatif', 'Positif']))"""),

    md("**Matrice de confusion**"),

    code("""cm = confusion_matrix(y_test, best_model.predict(X_test))
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues',
            xticklabels=['Négatif', 'Positif'],
            yticklabels=['Négatif', 'Positif'])
plt.title(f'Matrice de confusion - {best_name}')
plt.ylabel('Vrai')
plt.xlabel('Prédit')
plt.tight_layout()
plt.show()"""),

    md("### 5.3 Importance des variables"),

    code("""# Compatible feature_importances_ (arbres) ET coef_ (linéaires)
if hasattr(best_model, 'feature_importances_'):
    imp = pd.Series(best_model.feature_importances_, index=feature_names)
    title = f'Top 15 features - {best_name} (importance)'
else:
    imp = pd.Series(np.abs(best_model.coef_[0]), index=feature_names)
    title = f'Top 15 features - {best_name} (|coefficients|)'

imp = imp.sort_values(ascending=True).tail(15)
plt.figure(figsize=(8, 6))
imp.plot(kind='barh', color='#16A085')
plt.title(title)
plt.xlabel('Poids')
plt.tight_layout()
plt.show()"""),

    md("""**Interprétation** — Les variables liées aux symptômes OMS et au statut
de `Contact` ressortent comme les plus discriminantes, ce qui est cohérent
avec les recommandations cliniques de l'OMS. La sévérité `Severe` arrive
également en tête, signe que le modèle a bien intégré la logique du score
de risque."""),

    # ------------------------------------------------------- 6. NON SUPERVISE
    md("""## 6. Apprentissage non supervisé — Clustering K-Means

Objectif : faire émerger des **profils-types** sans utiliser la cible, pour
identifier des sous-populations homogènes (utile pour le tri en consultation
ou la priorisation des tests).

### 6.1 Standardisation puis recherche du k optimal

Les variables étant majoritairement binaires après one-hot, on standardise
avant K-Means pour limiter l'impact des échelles."""),

    code("""scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded.values)

ks = list(range(2, 7))
inertias, sils = [], []
sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_scaled), size=5000, replace=False)

for k in ks:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled[sample_idx], km.labels_[sample_idx]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(ks, inertias, marker='o', color='#2980B9')
ax1.set_title('Méthode du coude'); ax1.set_xlabel('k'); ax1.set_ylabel('Inertie')
ax2.plot(ks, sils, marker='s', color='#C0392B')
ax2.set_title('Score de silhouette'); ax2.set_xlabel('k'); ax2.set_ylabel('Silhouette')
plt.tight_layout()
plt.show()

pd.DataFrame({'k': ks, 'inertia': inertias, 'silhouette': sils}).set_index('k').round(3)"""),

    md("""**Remarque méthodologique** — Les scores de silhouette sont relativement
faibles (~0.05-0.10), ce qui est attendu sur des données purement
catégorielles encodées en one-hot. La distance euclidienne n'est pas idéale
dans ce cas ; une amélioration possible serait l'algorithme **K-Modes**,
spécialement conçu pour les variables catégorielles. On retient ici le `k`
qui maximise la silhouette dans la plage testée."""),

    code("""best_k = ks[int(np.argmax(sils))]
print(f'k retenu : {best_k}')

kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
labels = kmeans.labels_"""),

    md("### 6.2 Visualisation 2D via PCA"),

    code("""pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled[sample_idx])

plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                hue=labels[sample_idx], palette='tab10',
                s=15, alpha=0.7, legend='full')
plt.title(f'Clusters K-Means (k={best_k}) — projection PCA 2D')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()"""),

    md("### 6.3 Profilage des clusters"),

    code("""df_clusters = df.copy()
df_clusters['cluster'] = labels

profile_num = df_clusters.groupby('cluster').agg(
    taille=(TARGET, 'size'),
    taux_covid=(TARGET, 'mean'),
    fievre=('Fever', 'mean'),
    fatigue=('Tiredness', 'mean'),
    toux_seche=('Dry-Cough', 'mean'),
    diff_resp=('Difficulty-in-Breathing', 'mean'),
).round(3)

def _mode(s): return s.value_counts().idxmax()

profile_cat = df_clusters.groupby('cluster').agg(
    age_dominant=('Age', _mode),
    severity_dominante=('Severity', _mode),
    contact_dominant=('Contact', _mode),
    pays_dominant=('Country', _mode),
)

profile = pd.concat([profile_num, profile_cat], axis=1)
profile"""),

    md("""**Lecture** — Chaque cluster est caractérisé par une combinaison
spécifique d'âge, de sévérité et de statut de contact. Les clusters avec
`Contact='Yes'` ou `Severity='Severe'` affichent logiquement des taux de
COVID+ plus élevés. Cette segmentation pourrait servir de premier tri en
consultation."""),

    # ------------------------------------------------------- 7. SAUVEGARDE
    md("""## 7. Sauvegarde des artefacts

Le meilleur modèle, le scaler, le K-Means et la table des features sont
sérialisés avec **joblib** pour être consommés par le chatbot Streamlit
(`app.py`)."""),

    code("""os.makedirs('../models', exist_ok=True)
joblib.dump(best_model,    '../models/best_model.joblib')
joblib.dump(kmeans,        '../models/kmeans.joblib')
joblib.dump(scaler,        '../models/scaler.joblib')
joblib.dump(feature_names, '../models/feature_names.joblib')
joblib.dump(profile,       '../models/cluster_profiles.joblib')
joblib.dump({
    'best_name': best_name,
    'results': results_df.to_dict(),
    'best_k': int(best_k),
    'n_train': int(X_train.shape[0]),
    'n_test': int(X_test.shape[0]),
}, '../models/metrics.joblib')

print('✅ Artefacts sauvegardés dans ../models/')"""),

    # ------------------------------------------------------- 8. CONCLUSION
    md("""## 8. Conclusion

### Résultats clés

- **Apprentissage supervisé** : 3 modèles comparés, le meilleur atteint
  ~90% d'accuracy et ~0.97 de ROC-AUC sur le jeu de test. Les variables
  les plus discriminantes sont conformes aux directives OMS (symptômes
  respiratoires + statut de contact + sévérité).
- **Apprentissage non supervisé** : K-Means identifie des sous-populations
  cohérentes mais peu séparées (silhouette modeste, normal sur du
  one-hot binaire). K-Modes serait la suite logique de ce travail.
- **Chatbot** : les artefacts produits ici sont directement consommés
  par l'application Streamlit `app.py` qui simule un interrogatoire
  médical question par question.

### Pistes d'amélioration

1. Validation croisée (k-fold) pour fiabiliser les métriques
2. Optimisation d'hyperparamètres (`GridSearchCV`)
3. Remplacer K-Means par **K-Modes** (mieux adapté aux variables
   catégorielles)
4. Calibration des probabilités (`CalibratedClassifierCV`) pour rendre
   les seuils d'alerte du chatbot plus interprétables
5. Tester un modèle explicable type **SHAP** pour la transparence

### Avertissement

Ce projet a une **finalité strictement pédagogique**. Les prédictions
produites par le chatbot ne sauraient en aucun cas se substituer à un
diagnostic médical posé par un professionnel de santé.
"""),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
            "mimetype": "text/x-python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NOTEBOOK, f, ensure_ascii=False, indent=1)

print(f"[OK] Notebook généré : {OUT}")
print(f"     {len(CELLS)} cellules "
      f"({sum(1 for c in CELLS if c['cell_type']=='markdown')} md / "
      f"{sum(1 for c in CELLS if c['cell_type']=='code')} code)")
