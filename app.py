"""
app.py - Chatbot Streamlit
--------------------------
Interrogatoire médical interactif pour évaluer le risque COVID-19.

Lancer :
    streamlit run app.py

Le chatbot guide l'utilisateur question par question, comme une consultation,
puis combine :
    - une prédiction supervisée (probabilité de cas positif)
    - un placement dans un cluster K-Means (profil d'appartenance)
pour produire un avis indicatif (non médical).
"""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration de la page
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="COVID-19 Symptoms Checker",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")

# --------------------------------------------------------------------------- #
# Chargement des artefacts (mis en cache)
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
    profiles = joblib.load(os.path.join(MODELS_DIR, "cluster_profiles.joblib"))
    metrics = joblib.load(os.path.join(MODELS_DIR, "metrics.joblib"))
    return model, kmeans, scaler, feature_names, profiles, metrics


model, kmeans, scaler, feature_names, cluster_profiles, metrics = load_artifacts()

# --------------------------------------------------------------------------- #
# Modalités (doivent matcher generate_data.py)
# --------------------------------------------------------------------------- #
WHO_SYMPTOMS = {
    "Fever": "Fièvre",
    "Tiredness": "Fatigue",
    "Dry-Cough": "Toux sèche",
    "Difficulty-in-Breathing": "Difficulté à respirer",
    "Sore-Throat": "Mal de gorge",
}
OTHER_SYMPTOMS = {
    "Pains": "Douleurs musculaires",
    "Nasal-Congestion": "Congestion nasale",
    "Runny-Nose": "Écoulement nasal",
    "Diarrhea": "Diarrhée",
}
AGE_GROUPS = ["0-9", "10-19", "20-24", "25-59", "60+"]
GENDERS = ["Male", "Female", "Transgender"]
SEVERITY = ["None", "Mild", "Moderate", "Severe"]
SEVERITY_FR = {"None": "Aucune", "Mild": "Légère", "Moderate": "Modérée", "Severe": "Sévère"}
CONTACT = ["Yes", "No", "Dont-Know"]
CONTACT_FR = {"Yes": "Oui", "No": "Non", "Dont-Know": "Je ne sais pas"}
COUNTRIES = ["France", "Italy", "China", "USA", "Spain",
             "Germany", "UK", "India", "Iran", "Other-Country"]

# --------------------------------------------------------------------------- #
# State management
# --------------------------------------------------------------------------- #
DEFAULT_STATE = {
    "step": 0,
    "country": "France",
    "age": "25-59",
    "gender": "Male",
    "who_symptoms": [],
    "other_symptoms": [],
    "severity": "None",
    "contact": "No",
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

N_STEPS = 8  # 0 = intro, 1-7 = questions, 8 = résultat

def goto(step: int):
    st.session_state.step = step

def reset():
    for k, v in DEFAULT_STATE.items():
        st.session_state[k] = v

# --------------------------------------------------------------------------- #
# Sidebar (infos modèle + reset)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 🩺 COVID-19 Symptoms Checker")
    st.markdown("Outil d'aide à la décision conçu dans le cadre d'un projet "
                "M1 Data & IA. **N'a pas valeur de diagnostic médical.**")

    st.divider()
    st.markdown("**Modèle déployé**")
    st.code(metrics["best_name"])
    st.markdown(f"- F1-score : `{metrics['results']['F1-score'][metrics['best_name']]:.3f}`")
    st.markdown(f"- ROC-AUC  : `{metrics['results']['ROC-AUC'][metrics['best_name']]:.3f}`")
    st.markdown(f"- Train    : `{metrics['n_train']:,}` / Test : `{metrics['n_test']:,}`")
    st.markdown(f"- Clusters K-Means : `{metrics['best_k']}`")
    st.divider()
    if st.button("🔄 Recommencer l'interrogatoire", use_container_width=True):
        reset()
        st.rerun()

# --------------------------------------------------------------------------- #
# Barre de progression
# --------------------------------------------------------------------------- #
if 1 <= st.session_state.step <= 7:
    st.progress(st.session_state.step / 7,
                text=f"Question {st.session_state.step}/7")

# --------------------------------------------------------------------------- #
# Étape 0 — Accueil
# --------------------------------------------------------------------------- #
def render_intro():
    st.title("🩺 Interrogatoire médical COVID-19")
    st.markdown(
        """
        Bonjour 👋
        Je suis un assistant virtuel conçu pour aider à évaluer rapidement
        votre **risque potentiel de COVID-19** à partir de quelques questions
        standards, alignées sur les directives de l'**OMS** et du ministère
        indien de la Santé.

        L'interrogatoire se déroule en **7 questions courtes**.
        Vos réponses sont traitées localement par un modèle de Machine Learning
        entraîné sur 92 160 profils.

        ---
        > ⚠️ **Avertissement** — Les résultats fournis sont purement indicatifs
        > et ne constituent **en aucun cas un avis médical**.
        > En cas de doute, consultez un professionnel de santé.
        """
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("➡️ Commencer l'interrogatoire",
                  on_click=goto, args=(1,), type="primary",
                  use_container_width=True)

# --------------------------------------------------------------------------- #
# Étape 1 — Pays
# --------------------------------------------------------------------------- #
def render_q_country():
    st.subheader("🌍 Avez-vous récemment voyagé ou séjourné dans l'un de ces pays ?")
    st.caption("Plusieurs de ces pays ont été des foyers majeurs de l'épidémie.")
    st.session_state.country = st.selectbox(
        "Sélectionnez votre pays :", COUNTRIES,
        index=COUNTRIES.index(st.session_state.country),
        label_visibility="collapsed",
    )
    nav_buttons(prev_step=0, next_step=2)

# --------------------------------------------------------------------------- #
# Étape 2 — Âge + genre
# --------------------------------------------------------------------------- #
def render_q_demo():
    st.subheader("👤 Quelques informations personnelles")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tranche d'âge**")
        st.session_state.age = st.radio(
            "age", AGE_GROUPS,
            index=AGE_GROUPS.index(st.session_state.age),
            label_visibility="collapsed",
        )
    with c2:
        st.markdown("**Genre**")
        gender_fr = {"Male": "Homme", "Female": "Femme", "Transgender": "Autre"}
        st.session_state.gender = st.radio(
            "gender", GENDERS,
            index=GENDERS.index(st.session_state.gender),
            format_func=lambda g: gender_fr[g],
            label_visibility="collapsed",
        )
    nav_buttons(prev_step=1, next_step=3)

# --------------------------------------------------------------------------- #
# Étape 3 — Symptômes OMS
# --------------------------------------------------------------------------- #
def render_q_who_symptoms():
    st.subheader("🤒 Présentez-vous l'un des 5 symptômes principaux selon l'OMS ?")
    st.caption("Cochez tous les symptômes que vous ressentez actuellement.")
    selected = []
    for key, label in WHO_SYMPTOMS.items():
        checked = key in st.session_state.who_symptoms
        if st.checkbox(label, value=checked, key=f"who_{key}"):
            selected.append(key)
    st.session_state.who_symptoms = selected
    nav_buttons(prev_step=2, next_step=4)

# --------------------------------------------------------------------------- #
# Étape 4 — Autres symptômes
# --------------------------------------------------------------------------- #
def render_q_other_symptoms():
    st.subheader("🤧 Ressentez-vous d'autres symptômes ?")
    st.caption("Cochez toutes les réponses applicables.")
    selected = []
    for key, label in OTHER_SYMPTOMS.items():
        checked = key in st.session_state.other_symptoms
        if st.checkbox(label, value=checked, key=f"other_{key}"):
            selected.append(key)
    st.session_state.other_symptoms = selected
    nav_buttons(prev_step=3, next_step=5)

# --------------------------------------------------------------------------- #
# Étape 5 — Gravité
# --------------------------------------------------------------------------- #
def render_q_severity():
    st.subheader("📊 Comment évalueriez-vous la gravité globale de votre état ?")
    st.session_state.severity = st.radio(
        "severity", SEVERITY,
        index=SEVERITY.index(st.session_state.severity),
        format_func=lambda s: SEVERITY_FR[s],
        label_visibility="collapsed",
    )
    nav_buttons(prev_step=4, next_step=6)

# --------------------------------------------------------------------------- #
# Étape 6 — Contact
# --------------------------------------------------------------------------- #
def render_q_contact():
    st.subheader("🫂 Avez-vous été en contact avec une personne COVID-19 positive ?")
    st.session_state.contact = st.radio(
        "contact", CONTACT,
        index=CONTACT.index(st.session_state.contact),
        format_func=lambda c: CONTACT_FR[c],
        label_visibility="collapsed",
    )
    nav_buttons(prev_step=5, next_step=7)

# --------------------------------------------------------------------------- #
# Étape 7 — Récapitulatif
# --------------------------------------------------------------------------- #
def render_q_recap():
    st.subheader("✅ Récapitulatif de vos réponses")
    rows = [
        ("Pays", st.session_state.country),
        ("Âge", st.session_state.age),
        ("Genre", {"Male": "Homme", "Female": "Femme", "Transgender": "Autre"}[st.session_state.gender]),
        ("Symptômes OMS", ", ".join(WHO_SYMPTOMS[s] for s in st.session_state.who_symptoms) or "Aucun"),
        ("Autres symptômes", ", ".join(OTHER_SYMPTOMS[s] for s in st.session_state.other_symptoms) or "Aucun"),
        ("Gravité", SEVERITY_FR[st.session_state.severity]),
        ("Contact COVID", CONTACT_FR[st.session_state.contact]),
    ]
    st.table(pd.DataFrame(rows, columns=["Question", "Réponse"]).set_index("Question"))
    st.info("Confirmez pour obtenir l'analyse du modèle.")
    nav_buttons(prev_step=6, next_step=8, next_label="🔬 Lancer l'analyse")

# --------------------------------------------------------------------------- #
# Construction du vecteur de features
# --------------------------------------------------------------------------- #
def build_feature_vector() -> np.ndarray:
    """Construit le vecteur one-hot dans le bon ordre (feature_names)."""
    row = {f: 0 for f in feature_names}

    # Symptômes binaires
    for s in WHO_SYMPTOMS.keys():
        if s in feature_names:
            row[s] = 1 if s in st.session_state.who_symptoms else 0
    for s in OTHER_SYMPTOMS.keys():
        if s in feature_names:
            row[s] = 1 if s in st.session_state.other_symptoms else 0
    if "None_Experiencing" in feature_names:
        row["None_Experiencing"] = 1 if len(st.session_state.other_symptoms) == 0 else 0

    # Variables one-hot encodées
    one_hot_pairs = {
        "Age": st.session_state.age,
        "Gender": st.session_state.gender,
        "Severity": st.session_state.severity,
        "Contact": st.session_state.contact,
        "Country": st.session_state.country,
    }
    for var, val in one_hot_pairs.items():
        col = f"{var}_{val}"
        if col in row:
            row[col] = 1

    return np.array([[row[f] for f in feature_names]], dtype=float)

# --------------------------------------------------------------------------- #
# Étape 8 — Résultat
# --------------------------------------------------------------------------- #
def render_result():
    st.title("🔬 Analyse de vos réponses")

    X = build_feature_vector()
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    cluster_id = int(kmeans.predict(scaler.transform(X))[0])
    cluster_info = cluster_profiles.loc[cluster_id]

    # --- Carte principale : probabilité ---
    if proba >= 0.7:
        emoji, color, label = "🔴", "#E74C3C", "Risque élevé"
    elif proba >= 0.4:
        emoji, color, label = "🟠", "#F39C12", "Risque modéré"
    else:
        emoji, color, label = "🟢", "#27AE60", "Risque faible"

    st.markdown(
        f"""
        <div style="background-color: {color}22; padding: 1.5rem; border-radius: 12px;
                    border-left: 6px solid {color}; margin-bottom: 1.2rem;">
            <h2 style="margin:0; color: {color};">{emoji} {label}</h2>
            <p style="font-size: 1.1rem; margin: 0.6rem 0 0 0;">
                Probabilité estimée d'être positif :
                <b style="color: {color}; font-size: 1.3rem;">{proba * 100:.1f} %</b>
            </p>
            <p style="margin: 0.4rem 0 0 0; color: #555;">
                Prédiction binaire (seuil 50 %) :
                <b>{'Positif probable' if pred else 'Négatif probable'}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Cluster d'appartenance ---
    st.markdown("### 👥 Profil d'appartenance (clustering)")
    st.markdown(
        f"Selon une analyse non-supervisée (K-Means, k={metrics['best_k']}), "
        f"votre profil ressemble le plus au **cluster #{cluster_id}** :"
    )
    cluster_view = pd.DataFrame({
        "Caractéristique": [
            "Taille du cluster",
            "Taux de COVID+ observé",
            "Âge dominant",
            "Sévérité dominante",
            "Contact dominant",
            "Pays dominant",
        ],
        "Valeur": [
            f"{int(cluster_info['taille']):,} profils",
            f"{cluster_info['taux_covid'] * 100:.1f} %",
            cluster_info["age_dominant"],
            SEVERITY_FR.get(cluster_info["severity_dominante"], cluster_info["severity_dominante"]),
            CONTACT_FR.get(cluster_info["contact_dominant"], cluster_info["contact_dominant"]),
            cluster_info["pays_dominant"],
        ],
    })
    st.table(cluster_view.set_index("Caractéristique"))

    # --- Disclaimer ---
    st.warning(
        "⚠️ **Avis non-médical** — Ces résultats sont issus d'un modèle statistique "
        "à but pédagogique. Ils ne remplacent pas un diagnostic posé par un "
        "professionnel de santé. En cas de symptômes inquiétants, contactez "
        "votre médecin ou le 15 (SAMU) en France."
    )

    # --- Recommandations contextuelles ---
    with st.expander("💡 Recommandations indicatives"):
        if proba >= 0.7:
            st.markdown("""
            - **Consultez rapidement** un professionnel de santé.
            - Isolez-vous et portez un masque en présence d'autres personnes.
            - Réalisez un test de dépistage (RT-PCR ou antigénique).
            - Surveillez votre température et votre saturation en oxygène.
            """)
        elif proba >= 0.4:
            st.markdown("""
            - Limitez vos contacts par précaution.
            - Surveillez l'évolution des symptômes sur 48-72h.
            - Un test de dépistage est recommandé.
            - Consultez si les symptômes s'aggravent.
            """)
        else:
            st.markdown("""
            - Risque faible selon les symptômes déclarés.
            - Maintenez les gestes barrières habituels.
            - Reconsultez l'outil si de nouveaux symptômes apparaissent.
            """)

    # --- Navigation ---
    c1, c2 = st.columns(2)
    with c1:
        st.button("⬅️ Modifier mes réponses",
                  on_click=goto, args=(7,), use_container_width=True)
    with c2:
        if st.button("🔄 Nouvelle consultation", use_container_width=True, type="primary"):
            reset()
            st.rerun()

# --------------------------------------------------------------------------- #
# Boutons de navigation Précédent / Suivant
# --------------------------------------------------------------------------- #
def nav_buttons(prev_step: int, next_step: int, next_label: str = "Suivant ➡️"):
    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.button("⬅️ Précédent", on_click=goto, args=(prev_step,),
                  use_container_width=True)
    with c2:
        st.button(next_label, on_click=goto, args=(next_step,),
                  type="primary", use_container_width=True)

# --------------------------------------------------------------------------- #
# Routeur principal
# --------------------------------------------------------------------------- #
ROUTER = {
    0: render_intro,
    1: render_q_country,
    2: render_q_demo,
    3: render_q_who_symptoms,
    4: render_q_other_symptoms,
    5: render_q_severity,
    6: render_q_contact,
    7: render_q_recap,
    8: render_result,
}

ROUTER[st.session_state.step]()
