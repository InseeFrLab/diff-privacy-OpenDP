import streamlit as st
from app.initialisation import init_session_defaults
from fonctions import eps_from_rho_delta, rho_from_eps_delta, construire_dataframe_comparatif
from process_tools import process_request, process_request_dp
import opendp.prelude as dp
import numpy as np
import polars as pl
from math import log10
import plotly.express as px
import plotly.graph_objects as go
init_session_defaults()

dp.enable_features("contrib")

n = 100000
np.random.seed(42)

df = pl.DataFrame({
    "sexe": np.random.choice(["H", "F"], size=n),
    "region": np.random.choice(["Nord", "Sud"], size=n),
    "revenu_annuel": np.random.randint(10000, 80000, size=n),
    "profession": np.random.choice(["ingénieur", "médecin", "avocat"], size=n),
    "note_satisfaction": np.random.randint(0, 20, size=n),
    "heures_travaillees": np.random.randint(20, 60, size=n),
    "secteur d'activité": np.random.choice(["public", "privé", "associatif"], size=n)
})

context_param = {
    "data": df.lazy(),
    "privacy_unit": dp.unit_of(contributions=1),
    "margins": [
        dp.polars.Margin(
            max_partition_length=1000
        ),
        dp.polars.Margin(
            by=["secteur d'activité", "region", "sexe", "profession"],
            public_info="keys",
        ),
    ],
}

# Définir les valeurs possibles pour chaque clé
key_values = {
    "sexe": ["H", "F"],
    "region": ["Nord", "Sud"],
    "profession": ["ingénieur", "médecin", "avocat"],
    "secteur d'activité": ["public", "privé", "associatif"]
}

requetes = st.session_state.data_json
nb_req = len(requetes)

nb_req_count = sum(1 for req in requetes.values() if req.get("type") == "count")
nb_req_sum = sum(1 for req in requetes.values() if req.get("type") == "sum")
nb_req_mean = sum(1 for req in requetes.values() if req.get("type") == "mean")
nb_req_quantile = sum(1 for req in requetes.values() if req.get("type") == "quantile")

# Poids globaux par type (normalisés)
if nb_req - nb_req_quantile > 0:
    poids_par_type = {
        "count": st.session_state.poids_count / nb_req_count,
        "sum": st.session_state.poids_sum / nb_req_sum,
        "mean": st.session_state.poids_mean / nb_req_mean
    }
else:
    poids_par_type = {"count": 0.0, "sum": 0.0, "mean": 0.0}

# Liste des poids, dans l’ordre des requêtes (hors quantile)
poids_requetes_rho = [
    poids_par_type[req["type"]]
    for req in requetes.values()
    if req.get("type") in ["count", "sum", "mean"]
]


# Sidebar
st.sidebar.header("Budget de confidentialité défini")
eps_tot = st.sidebar.slider(r"Epsilon $\varepsilon$", 0.1, 10.0, value=st.session_state.eps_tot, step=0.1)
delta_exp = st.sidebar.slider(r"Delta $\delta = 10^{x}$", -10, -1, value=int(log10(st.session_state.delta_tot)))
delta_tot = 10 ** (delta_exp)
# Initialisation de la barre dans la sidebar
progress_bar = st.sidebar.progress(0, text="Progression du traitement des requêtes...")

rho_utilise = rho_from_eps_delta(eps_tot, delta_tot) * (1 - st.session_state.poids_quantile)
eps_rest = eps_tot - eps_from_rho_delta(rho_utilise, delta_tot)

context_rho = dp.Context.compositor(
        **context_param,
        privacy_loss=dp.loss_of(rho=rho_utilise),
        split_by_weights=poids_requetes_rho
    )

context_eps = dp.Context.compositor(
        **context_param,
        privacy_loss=dp.loss_of(epsilon=eps_rest),
        split_evenly_over=nb_req_quantile
    )

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
    print(df_plot)

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
