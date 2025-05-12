import streamlit as st
from app.initialisation import init_session_defaults
from fonctions import eps_from_rho_delta, rho_from_eps_delta
from process_tools import process_request, process_request_dp
import opendp.prelude as dp
import numpy as np
import polars as pl
from math import log10
init_session_defaults()

dp.enable_features("contrib")

n = 1000
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
eps_tot = st.sidebar.slider("𝜖 Epsilon", 0.1, 10.0, value=st.session_state.eps_tot, step=0.1)
delta_exp = st.sidebar.slider("🔽 -log₁₀(δ)", 1, 10, value=int(-log10(st.session_state.delta_tot)))
delta_tot = 10 ** (-delta_exp)
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

tab1, tab2 = st.tabs(["Résultats", "Précision"])

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
