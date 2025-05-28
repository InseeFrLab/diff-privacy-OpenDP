import streamlit as st
from app.initialisation import init_session_defaults, update_context
from src.fonctions import construire_dataframe_comparatif
from src.process_tools import process_request, process_request_dp
import opendp.prelude as dp
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
init_session_defaults()

dp.enable_features("contrib")

df = st.session_state.df
context_param = st.session_state.context_param
key_values = st.session_state.key_values

requetes = st.session_state.requetes
nb_req = len(requetes)

# Sidebar
st.sidebar.header("Budget de confidentialité défini")
rho_budget = st.sidebar.slider(r"Budget $\rho$ de l'étude", 0.01, 1.0, value=st.session_state.rho_budget)
# Initialisation de la barre dans la sidebar
progress_bar = st.sidebar.progress(0, text="Progression du traitement des requêtes...")

(context_rho, context_eps) = update_context(rho_budget, st.session_state.poids_requetes_rho, st.session_state.poids_requetes_quantile)

tab1, tab2, tab3, tab4 = st.tabs(["Résultats", "Précision", "Tableau résumé", "Dataviz"])

# Dictionnaire pour stocker les résultats DP
resultats_dp = {}

# Dictionnaire pour stocker les résultats réels
resultats_reels = {}

# --- Calcul (1 seule fois) ---
for i, (key, req) in enumerate(requetes.items(), 1):
    try:
        # Résultat réel
        resultat = process_request(df.lazy(), req)
        if req.get("by") is not None:
            resultat = resultat.sort(by=req.get("by"))
        resultats_reels[key] = resultat
    except Exception as e:
        resultats_reels[key] = e

    try:
        # Résultat DP
        resultat_dp = process_request_dp(context_rho, context_eps, key_values, req)
        df_result = resultat_dp.release().collect()
        if req.get("by") is not None:
            df_result = df_result.sort(by=req.get("by"))
            first_col = df_result.columns[0]
            new_order = df_result.columns[1:] + [first_col]
            df_result = df_result.select(new_order)
        resultats_dp[key] = {"data": df_result, "summary": resultat_dp.summarize(alpha=0.05)}
    except Exception as e:
        resultats_dp[key] = {"data": e, "summary": e}

    progress_bar.progress(int(100 * i / nb_req), text=f"Progression : {i}/{nb_req}")

# --- Affichage ---
with tab1:
    col_reel, col_dp = st.columns(2)
    for key in requetes:
        with col_reel:
            st.markdown(f"### ✅ Résultat réel pour : `{key}`")
            res = resultats_reels[key]
            if isinstance(res, Exception):
                st.error(f"Erreur : {res}")
            else:
                st.dataframe(res)

        with col_dp:
            st.markdown(f"### 🔐 Résultat DP pour : `{key}`")
            res = resultats_dp[key]["data"]
            if isinstance(res, Exception):
                st.error(f"Erreur : {res}")
            else:
                st.dataframe(res)

with tab2:
    for key in requetes:
        st.markdown(f"### 📏 Précision à 95% pour : `{key}`")
        res = resultats_dp[key]["summary"]
        if isinstance(res, Exception):
            st.error(f"Erreur : {res}")
        else:
            st.dataframe(res)

df_comparatif = construire_dataframe_comparatif(resultats_reels, resultats_dp, requetes)

with tab3:
    st.markdown("### 📊 Comparaison des résultats réels et DP")
    st.dataframe(df_comparatif)

with tab4:
    # Conversion du DataFrame Polars → Pandas
    df_plot = df_comparatif.to_pandas()

    # Créer la colonne scale_size
    def compute_scale_size(row):
        # On exclut quantiles et moyennes
        if str(row["type_requête"]).startswith("quantile") or str(row["type_requête"]).startswith("mean"):
            return np.nan
        # Calcul du scale_size pour les autres
        else:
            return row["scale"][0] / row["scale"][1] if len(row["scale"]) > 1 else row["scale"][0]
        return np.nan

    df_plot["scale_size"] = df_plot.apply(compute_scale_size, axis=1)
    # print(df_plot)

    df_plot = df_plot.dropna(subset=["valeur_réelle", "valeur_DP"])

    # Création du graphique
    fig = go.Figure()

    # Couleurs automatiques par type de requête
    type_requetes = df_plot["type_requête"].unique()
    color_map = px.colors.qualitative.Plotly

    for i, t in enumerate(type_requetes):
        df_type = df_plot[df_plot["type_requête"] == t]
        color = color_map[i % len(color_map)]

        # Ajouter les cercles si scale_size est défini
        df_with_scale = df_type[df_type["scale_size"].notna()]
        if not df_with_scale.empty:
            fig.add_trace(go.Scatter(
                x=df_with_scale["valeur_réelle"],
                y=df_with_scale["valeur_DP"],
                mode="markers",
                marker=dict(
                    size=2 * df_with_scale["scale_size"],
                    color=color,
                    opacity=0.2,
                    line=dict(width=0)
                ),
                name=f"{t} (incertitude)",
                showlegend=True,
                hoverinfo="skip"
            ))

        # Ajouter les points principaux (toujours)
        fig.add_trace(go.Scatter(
            x=df_type["valeur_réelle"],
            y=df_type["valeur_DP"],
            mode="markers",
            marker=dict(
                size=6,
                color=color,
                line=dict(width=1, color="black")
            ),
            name=t
        ))

    # Ajouter la droite y = x
    global_min = min(df_plot["valeur_réelle"].min(), df_plot["valeur_DP"].min())
    global_max = max(df_plot["valeur_réelle"].max(), df_plot["valeur_DP"].max())

    fig.add_trace(go.Scatter(
        x=[global_min, global_max],
        y=[global_min, global_max],
        mode="lines",
        line=dict(color="black", dash="dash"),
        name="y = x"
    ))

    # Mise en page : quadrillage, fond blanc
    fig.update_layout(
        title="Comparaison des valeurs réelles vs bruitées avec incertitude (scale)",
        xaxis=dict(
            title="Valeur réelle",
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False
        ),
        yaxis=dict(
            title="Valeur bruitée (DP)",
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False
        ),
        legend_title="Type de requête",
        plot_bgcolor="white"
    )

    # Affichage Streamlit
    st.plotly_chart(fig)

    # Étape 2 : Agrégation (exemple : moyenne par requête)
    df_agg = df_plot.groupby("requête", as_index=False)["écart_relatif"].mean()
    df_agg = df_agg.merge(df_plot.drop_duplicates(subset=["requête"])[["requête", "type_requête"]], on="requête", how="left")

    # Étape 3 : Barplot
    fig_bar = px.bar(
        df_agg,
        x="requête",
        y="écart_relatif",
        title="Écart relatif moyen par requête",
        labels={"écart_relatif": "Écart relatif moyen", "requête": "Requête"},
        text_auto=".2%",  # Affiche les valeurs sur les barres
        color="type_requête",
    )

    fig_bar.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Écart relatif moyen", showgrid=True, gridcolor='lightgrey'),
        plot_bgcolor="white"
    )

    # Affichage dans Streamlit
    st.plotly_chart(fig_bar)
