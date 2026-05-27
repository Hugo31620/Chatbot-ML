# 🩺 COVID-19 Symptoms Checker

> **Projet M1 Data & IA — YNOV**
> Chatbot d'interrogatoire médical assisté par Machine Learning, conçu pour
> aider à évaluer rapidement le risque potentiel de COVID-19 à partir de
> symptômes standards (directives **OMS** et ministère indien de la Santé).

---

## 📋 Sommaire

1. [Objectif](#-objectif)
2. [Architecture du projet](#-architecture-du-projet)
3. [Installation](#-installation)
4. [Utilisation](#-utilisation)
5. [Méthodologie](#-méthodologie)
6. [Résultats](#-résultats)
7. [Démo & présentation orale](#-démo--présentation-orale)
8. [Disclaimer](#-disclaimer)

---

## 🎯 Objectif

Construire un **chatbot médical** qui :

- mène un **interrogatoire question par question** (pays visité → âge →
  symptômes OMS → autres symptômes → gravité → contact),
- combine une **prédiction supervisée** (probabilité d'être positif) avec
  un **profilage non supervisé** (cluster d'appartenance),
- aide le professionnel de santé à **trier plus vite** les patients à risque.

Le projet couvre les **deux familles d'algorithmes** demandées :

| Type | Algorithme | Rôle |
|------|-----------|------|
| **Supervisé** | Logistic Regression / Random Forest / Gradient Boosting | Prédire `Covid-Positive` (0/1) |
| **Non supervisé** | K-Means (+ PCA) | Identifier des profils-types de patients |

---

## 🗂 Architecture du projet

```
covid-chatbot/
├── app.py                       ← Chatbot Streamlit (entrée utilisateur)
├── train_pipeline.py            ← Pipeline ML complet (EDA + train + save)
├── build_notebook.py            ← Génère le notebook d'analyse
│
├── data/
│   ├── generate_data.py         ← Génération du dataset selon les specs OMS
│   ├── Raw-Data.csv             ← Modalités possibles de chaque variable
│   └── Cleaned-Data.csv         ← 92 160 combinaisons + cible (généré)
│
├── notebooks/
│   └── 01_analyse_covid.ipynb   ← Notebook complet d'analyse
│
├── models/                      ← Artefacts ML sérialisés (générés)
│   ├── best_model.joblib
│   ├── kmeans.joblib
│   ├── scaler.joblib
│   ├── feature_names.joblib
│   ├── cluster_profiles.joblib
│   └── metrics.joblib
│
├── assets/                      ← Graphiques générés par le pipeline
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── models_comparison.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── elbow_silhouette.png
│   ├── pca_clusters.png
│   └── symptoms_distribution.png
│
├── requirements.txt
└── README.md
```

---

## 🔧 Installation

### Prérequis
- Python ≥ 3.10
- pip

### Étapes

```bash
# 1. Cloner / extraire le projet
cd covid-chatbot

# 2. (Recommandé) créer un environnement virtuel
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Utilisation

### Pipeline complet en 3 commandes

```bash
# 1) Générer le dataset (Raw-Data.csv + Cleaned-Data.csv)
python data/generate_data.py

# 2) Entraîner tous les modèles + générer les graphiques
python train_pipeline.py

# 3) Lancer le chatbot
streamlit run app.py
```

L'application s'ouvre automatiquement dans le navigateur sur
`http://localhost:8501`.

### (Optionnel) Explorer le notebook d'analyse

```bash
python build_notebook.py                # (re)génère le .ipynb
jupyter notebook notebooks/01_analyse_covid.ipynb
```

---

## 🧪 Méthodologie

### 1. Données

- **Source** : structure issue du [dataset Kaggle COVID-19 Symptoms
  Checker](https://www.kaggle.com/iamhungundji/covid19-symptoms-checker).
- **Génération locale** : le script `data/generate_data.py` construit
  l'intégralité des combinaisons possibles des 7 variables catégorielles
  (≈92 160 lignes), avec une cible binaire `Covid-Positive` dérivée d'un
  **score de risque pondéré** inspiré des facteurs OMS :

  ```
  score = 2 × Σ(symptômes OMS)  +  1 × Σ(autres symptômes)
        + 3 (Contact=Yes) ou 1 (Dont-Know)
        + 3 (Severity=Severe) ou 1 (Moderate) ou 0.5 (Mild)
        + 1 (Age=60+) + 1 (Pays foyer majeur)
        + bruit gaussien
  → binarisation autour de la médiane
  ```

- **Variables** :

  | Variable | Modalités |
  |----------|-----------|
  | `Country` | France, Italy, China, USA, Spain, Germany, UK, India, Iran, Other-Country |
  | `Age` | 0-9, 10-19, 20-24, 25-59, 60+ |
  | `Gender` | Male, Female, Transgender |
  | `Symptômes OMS` (binaires) | Fever, Tiredness, Dry-Cough, Difficulty-in-Breathing, Sore-Throat |
  | `Autres symptômes` (binaires) | Pains, Nasal-Congestion, Runny-Nose, Diarrhea, None_Experiencing |
  | `Severity` | None, Mild, Moderate, Severe |
  | `Contact` | Yes, No, Dont-Know |

### 2. Préprocessing

- Variables binaires : conservées telles quelles
- Variables catégorielles : **one-hot encoding** (`pd.get_dummies`)
- Split train/test : **80/20 stratifié**
- Standardisation (`StandardScaler`) **uniquement pour le clustering**

### 3. Apprentissage supervisé

Trois modèles complémentaires sont comparés :

| Modèle | Famille | Force principale |
|--------|---------|------------------|
| **Logistic Regression** | Linéaire | Interprétabilité, baseline solide |
| **Random Forest** | Bagging d'arbres | Non-linéarités, robustesse |
| **Gradient Boosting** | Boosting d'arbres | Performance fine sur tabulaire |

Métriques : *accuracy, precision, recall, F1-score, ROC-AUC*.
**Sélection sur F1** (équilibre précision/rappel important en contexte
médical).

### 4. Apprentissage non supervisé

- **K-Means** avec recherche de `k` optimal sur la plage `[2, 6]`
- Critères : **méthode du coude** (inertie) + **silhouette** (sur échantillon
  de 5 000 points pour le coût)
- Visualisation 2D via **PCA**
- **Profilage** : pour chaque cluster, on extrait :
  - taille, taux de COVID+ observé,
  - moyennes des symptômes OMS,
  - modalité dominante d'âge / sévérité / contact / pays.

### 5. Intégration dans le chatbot

L'app Streamlit charge tous les artefacts (`models/*.joblib`) et expose un
parcours en **7 questions** + page de résultat. La page finale affiche :

- la **probabilité** estimée d'être positif (avec code couleur 🟢🟠🔴),
- le **cluster d'appartenance** et son profil typique,
- des **recommandations indicatives** adaptées au niveau de risque,
- le **disclaimer médical**.

---

## 📊 Résultats

### Apprentissage supervisé (test set, 18 432 profils)

| Modèle | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| LogisticRegression | **0.901** | **0.921** | 0.881 | **0.899** | **0.969** |
| RandomForest       | 0.883 | 0.884 | 0.882 | 0.883 | 0.958 |
| GradientBoosting   | 0.898 | 0.904 | 0.892 | 0.897 | 0.967 |

→ **Modèle retenu : Logistic Regression** (meilleur F1, très interprétable).

### Apprentissage non supervisé

- **k retenu : 6** (max de silhouette sur `[2, 6]`)
- Clusters bien différenciés sur les **variables catégorielles**
  (Age × Severity × Contact), moins sur les symptômes (effet du one-hot
  binaire — voir note méthodologique dans le notebook).

---

## 🎤 Démo & présentation orale

### Déroulé suggéré (10 min)

| Temps | Section | Support |
|-------|---------|---------|
| 0:00–1:00 | Contexte & objectifs | README §1 |
| 1:00–3:00 | EDA & insights | Notebook §3 (graphiques générés) |
| 3:00–5:00 | Comparaison des modèles supervisés | Notebook §5 + `models_comparison.png` |
| 5:00–6:30 | Clustering K-Means + profilage | Notebook §6 + `pca_clusters.png` |
| 6:30–9:00 | **Démo live du chatbot** | `streamlit run app.py` |
| 9:00–10:00 | Limites & pistes d'amélioration | Notebook §8 |

### Conseils

- **Préparer deux profils contrastés** à dérouler pendant la démo :
  1. *Cas à risque* : 60+, France, fièvre + toux + difficulté à respirer,
     Severity=Severe, Contact=Yes → doit donner ~99% de probabilité.
  2. *Cas sain* : 25-59, autre pays, aucun symptôme, Contact=No → doit
     donner ~0%.
- Mettre en avant la **double approche** supervisée + non-supervisée : le
  jury de M1 valorise la complémentarité.
- Anticiper les questions sur la **silhouette faible** → réponse :
  *« données purement catégorielles, K-Modes serait plus adapté, on a
  retenu K-Means pour l'interprétabilité et la cohérence avec le
  programme ».*

---

## ⚠️ Disclaimer

Ce projet est **strictement pédagogique**. Les prédictions du chatbot
**ne constituent en aucun cas un diagnostic médical**. En cas de symptômes
inquiétants, contactez un professionnel de santé ou le 15 (SAMU France).

---

## 📚 Sources

- [OMS — Symptômes du COVID-19](https://www.who.int/health-topics/coronavirus)
- [Ministère indien de la Santé](https://www.mohfw.gov.in/)
- [Dataset Kaggle](https://www.kaggle.com/iamhungundji/covid19-symptoms-checker)

---

*Réalisé dans le cadre du Master 1 Data & IA — YNOV.*
