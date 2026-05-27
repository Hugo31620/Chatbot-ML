import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(
    page_title="COVID-19 — Dashboard ML & Cartographie",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS : masquer le header Streamlit et les paddings pour plein écran ────────
st.markdown("""
<style>
    /* Masque le header et le footer Streamlit */
    #MainMenu, header, footer { visibility: hidden; height: 0; }
    
    /* Supprime les paddings autour du composant */
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    
    /* Titre de la page dans la sidebar */
    section[data-testid="stSidebar"] { background: #111827; }

    /* Retire le fond blanc par défaut de l'iframe */
    iframe { border: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Chargement du HTML ────────────────────────────────────────────────────────
HTML_FILE = os.path.join(os.path.dirname(__file__), "..", "covid19_france.html")

@st.cache_resource
def load_dashboard_html():
    """Charge le fichier HTML une seule fois (mis en cache)."""
    try:
        with open(HTML_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None

html_content = load_dashboard_html()

# ── Affichage ─────────────────────────────────────────────────────────────────
if html_content:
    # Calcul dynamique de la hauteur de la fenêtre via JS
    components.html(
        html_content,
        height=920,
        scrolling=True,
    )
else:
    st.error("⚠️ Fichier `covid19_france.html` introuvable.")
    st.markdown("""
    **Comment corriger :**
    1. Placez `covid19_france.html` à la **racine** de votre dépôt GitHub
    2. Votre arborescence doit ressembler à :
    ```
    ├── app.py               ← votre app principale
    ├── covid19_france.html  ← ici
    └── pages/
        └── 2_COVID19_Dashboard.py  ← ce fichier
    ```
    3. Redéployez sur Streamlit Cloud
    """)
