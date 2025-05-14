import streamlit as st
import json
import plotly.express as px
from app.initialisation import init_session_defaults, update_info_request
init_session_defaults()

st.title("🧾 Éditeur de requêtes JSON")

# --- Sidebar ---
st.sidebar.header("📂 Chargement JSON")
uploaded_file = st.sidebar.file_uploader("Chargez un fichier .json", type="json")

if st.sidebar.button("🔄 Réinitialiser avec `data_example.json`"):
    try:
        with open("data_example.json", "r", encoding="utf-8") as f:
            st.session_state.requetes = json.load(f)
            st.session_state.source = "📦 Fichier par défaut : `data_example.json`"
        st.sidebar.success("✅ Réinitialisé avec succès.")
    except FileNotFoundError:
        st.sidebar.error("❌ `data_example.json` introuvable.")

if uploaded_file:
    try:
        st.session_state.requetes = json.load(uploaded_file)
        st.session_state.source = "📁 Fichier importé"
        st.sidebar.success("✅ Fichier chargé.")
    except json.JSONDecodeError:
        st.sidebar.error("❌ JSON invalide. Rechargement ignoré.")

st.sidebar.markdown("💡 Par défaut, le fichier `data_example.json` est utilisé.")

# --- Affichage JSON actuel ---
st.subheader("📋 Requêtes actuelles")

if st.session_state.requetes:
    st.caption(st.session_state.source)
    st.json(st.session_state.requetes, expanded=True)
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
    requetes = st.session_state.requetes
    # Génère un identifiant unique du type req_1, req_2, ...
    existing_keys = set(st.session_state.requetes.keys())
    i = 1
    while f"req_{i}" in existing_keys:
        i += 1
    new_key = f"req_{i}"
    try:
        bounds_val = json.loads(bounds) if bounds else None

        requetes[new_key] = {
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
        update_info_request()
        st.rerun()

# --- Suppression d'une requête ---
st.subheader("🗑️ Supprimer une requête")

if st.session_state.requetes:
    req_list = list(st.session_state.requetes.keys())
    selected_req = st.selectbox("Sélectionnez une requête à supprimer", req_list)

    if st.button("❌ Supprimer la requête"):
        del st.session_state.requetes[selected_req]
        st.session_state["success_message"] = f"🗑️ Requête `{selected_req}` supprimée."
        update_info_request()
        st.rerun()
else:
    st.info("Aucune requête disponible à supprimer.")

# --- Comptage des types de requêtes ---
requetes = st.session_state.requetes
nb_req = len(requetes)

nb_req_count = st.session_state.nb_req_count
nb_req_sum = st.session_state.nb_req_sum
nb_req_mean = st.session_state.nb_req_mean
nb_req_quantile = st.session_state.nb_req_quantile

# --- Affichage structuré ---
st.title("📊 Informations sur les requêtes à traiter")
st.write(f"Nombre total de requêtes : **{nb_req}**")

col1, col2, col3, col4 = st.columns(4)
col1.metric("🔢 Comptage", nb_req_count)
col2.metric("➕ Somme", nb_req_sum)
col3.metric("📏 Moyenne", nb_req_mean)
col4.metric("📊 Quantile", nb_req_quantile)

# --- Visualisation par graphique circulaire ---
data_pie = {
    "Type": ["count", "sum", "mean", "quantile"],
    "Nombre": [nb_req_count, nb_req_sum, nb_req_mean, nb_req_quantile]
}
fig = px.pie(data_pie, names="Type", values="Nombre", title="Répartition des types de requêtes")
st.plotly_chart(fig, use_container_width=True)
