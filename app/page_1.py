# app.py
import streamlit as st
import numpy as np
import polars as pl
import plotly.express as px

# Génération des données simulées
n = 1000
np.random.seed(42)

df_polars = pl.DataFrame({
    "sexe": np.random.choice(["H", "F"], size=n),
    "region": np.random.choice(["Nord", "Sud"], size=n),
    "revenu_annuel": np.random.randint(10000, 80000, size=n),
    "profession": np.random.choice(["ingénieur", "médecin", "avocat"], size=n),
    "note_satisfaction": np.random.randint(0, 20, size=n),
    "heures_travaillees": np.random.randint(20, 60, size=n),
    "secteur d'activité": np.random.choice(["public", "privé", "associatif"], size=n)
})

# Conversion en pandas
df = df_polars.to_pandas()


# Sidebar
st.sidebar.header("🎛️ Filtres")
sexe = st.sidebar.selectbox("Sexe :", options=["Tous"] + df["sexe"].unique().tolist())
region = st.sidebar.selectbox("Région :", options=["Toutes"] + df["region"].unique().tolist())
revenu_min, revenu_max = st.sidebar.slider("Revenu annuel :",
                                           int(df["revenu_annuel"].min()),
                                           int(df["revenu_annuel"].max()),
                                           (int(df["revenu_annuel"].min()), int(df["revenu_annuel"].max())),
                                           step=1000)
refresh = st.sidebar.button("🔁 Actualiser")

# --- Données filtrées ---
df_filtered = df.copy()

if sexe != "Tous":
    df_filtered = df_filtered[df_filtered["sexe"] == sexe]

if region != "Toutes":
    df_filtered = df_filtered[df_filtered["region"] == region]

df_filtered = df_filtered[
    (df_filtered["revenu_annuel"] >= revenu_min) &
    (df_filtered["revenu_annuel"] <= revenu_max)
]

# --- Onglets ---
tab1, tab2, tab3 = st.tabs(["📋 Données", "📈 Diagramme de dispersion", "🧮 Statistiques descriptives"])

with tab1:
    st.subheader("Données filtrées")
    st.dataframe(df_filtered, use_container_width=True)

with tab2:
    st.subheader("Satisfaction vs Heures travaillées")
    if df_filtered.empty:
        st.warning("Aucune donnée à afficher.")
    else:
        fig = px.scatter(df_filtered,
                         x="heures_travaillees",
                         y="note_satisfaction",
                         color="profession",
                         title="Satisfaction vs Heures travaillées",
                         labels={"heures_travaillees": "Heures travaillées",
                                 "note_satisfaction": "Note de satisfaction"},
                         template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Résumé statistique")
    if df_filtered.empty:
        st.info("Aucune donnée à résumer.")
    else:
        st.text(df_filtered.describe().round(2).to_string())
