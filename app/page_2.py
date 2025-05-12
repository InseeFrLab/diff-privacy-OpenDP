import streamlit as st
import json
from app.initialisation import init_session_defaults
init_session_defaults()

st.title("🧾 Éditeur de requêtes JSON")

# --- Sidebar ---
st.sidebar.header("📂 Chargement JSON")
uploaded_file = st.sidebar.file_uploader("Chargez un fichier .json", type="json")

if st.sidebar.button("🔄 Réinitialiser avec `data_example.json`"):
    try:
        with open("data_example.json", "r", encoding="utf-8") as f:
            st.session_state.data_json = json.load(f)
            st.session_state.source = "📦 Fichier par défaut : `data_example.json`"
        st.sidebar.success("✅ Réinitialisé avec succès.")
    except FileNotFoundError:
        st.sidebar.error("❌ `data_example.json` introuvable.")

if uploaded_file:
    try:
        st.session_state.data_json = json.load(uploaded_file)
        st.session_state.source = "📁 Fichier importé"
        st.sidebar.success("✅ Fichier chargé.")
    except json.JSONDecodeError:
        st.sidebar.error("❌ JSON invalide. Rechargement ignoré.")

st.sidebar.markdown("💡 Par défaut, le fichier `data_example.json` est utilisé.")

# --- Affichage JSON actuel ---
st.subheader("📋 Requêtes actuelles")

if st.session_state.data_json:
    st.caption(st.session_state.source)
    st.json(st.session_state.data_json, expanded=True)
else:
    st.info("Aucune requête à afficher.")

if "success_message" in st.session_state:
    st.success(st.session_state.success_message)
    del st.session_state.success_message

# --- Formulaire d'ajout ---
st.subheader("➕ Ajouter une requête")

with st.form("form_ajout_requete", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        req_type = st.selectbox("Type", options=["count", "sum", "mean", "quantile"])
        variable = st.text_input("Variable (laisser vide si 'count')", value="")
        bounds = st.text_input("Bornes (ex: [0, 100])", value="")
    with col2:
        by = st.text_input("Regrouper par (séparé par des virgules)", value="")
        filtre = st.text_input("Filtre (ex: heures_travaillees > 50)", value="")

    submit = st.form_submit_button("Ajouter la requête")

if submit:
    data = st.session_state.data_json
    # Génère un identifiant unique du type req_1, req_2, ...
    existing_keys = set(st.session_state.data_json.keys())
    i = 1
    while f"req_{i}" in existing_keys:
        i += 1
    new_key = f"req_{i}"
    try:
        bounds_val = json.loads(bounds) if bounds else None

        data[new_key] = {
            "type": req_type,
            "variable": variable or None,
            "bounds": bounds_val,
            "by": [s.strip() for s in by.split(",")] if by else None,
            "filtre": filtre or None
        }

    except json.JSONDecodeError:
        st.error("⚠️ Format des bornes invalide. Utilisez un tableau JSON valide (ex: [0, 100]).")
        bounds_val = None

    else:
        st.session_state["success_message"] = f"✅ Requête {new_key} ajoutée."
        st.rerun()

# --- Suppression d'une requête ---
st.subheader("🗑️ Supprimer une requête")

if st.session_state.data_json:
    req_list = list(st.session_state.data_json.keys())
    selected_req = st.selectbox("Sélectionnez une requête à supprimer", req_list)

    if st.button("❌ Supprimer la requête"):
        del st.session_state.data_json[selected_req]
        st.session_state["success_message"] = f"🗑️ Requête `{selected_req}` supprimée."
        st.rerun()
else:
    st.info("Aucune requête disponible à supprimer.")
