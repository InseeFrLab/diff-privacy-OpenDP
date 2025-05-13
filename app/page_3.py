import streamlit as st
import plotly.express as px
from app.initialisation import init_session_defaults
from fonctions import eps_from_rho_delta, rho_from_eps_delta
from process_tools import process_request_dp
import opendp.prelude as dp
import numpy as np
import polars as pl
init_session_defaults()

# --- Comptage des types de requêtes ---
requetes = st.session_state.data_json
nb_req = len(requetes)

nb_req_count = sum(1 for req in requetes.values() if req.get("type") == "count")
nb_req_sum = sum(1 for req in requetes.values() if req.get("type") == "sum")
nb_req_mean = sum(1 for req in requetes.values() if req.get("type") == "mean")
nb_req_quantile = sum(1 for req in requetes.values() if req.get("type") == "quantile")

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

st.subheader("🔐 Paramètres de confidentialité")

col1, col2, col3 = st.columns(3)
eps_tot = col1.slider(r"Epsilon $\varepsilon$", 0.1, 10.0, 3.0, step=0.1)
delta_exp = col2.slider(r"Delta $\delta = 10^{x}$", -10, -1, -5)
delta_tot = 10 ** (delta_exp)

# --- Calcul et affichage de rho ---
rho_tot = rho_from_eps_delta(eps_tot, delta_tot)
col3.metric(label=r"Budget équivalent en Rho $\rho$", value=f"{rho_tot:.4f}")

st.subheader("⚖️ Répartition du budget (normalisé automatiquement)")

col1, col2, col3, col4 = st.columns(4)
poids_count_raw = col1.slider("Poids des comptages dans la part du budget en rho", 0., 1., nb_req_count / nb_req, step=0.05)
poids_sum_raw = col2.slider("Poids des sommes dans la part du budget en rho", 0., 1., nb_req_sum / nb_req, step=0.05)
poids_mean_raw = col3.slider("Poids des moyennes dans la part du budget en rho", 0., 1., nb_req_mean / nb_req, step=0.05)
poids_quantile_raw = col4.slider("Poids des quantiles dans la part du budget en rho", 0., 1., nb_req_quantile / nb_req, step=0.05)

# --- Normalisation automatique ---
total_raw = poids_count_raw + poids_sum_raw + poids_mean_raw + poids_quantile_raw

if total_raw == 0:
    st.warning("⚠️ La somme des poids est nulle. Les poids ne peuvent pas être normalisés.")
    poids_count = poids_sum = poids_mean = poids_quantile = 0
else:
    poids_count = poids_count_raw / total_raw
    poids_sum = poids_sum_raw / total_raw
    poids_mean = poids_mean_raw / total_raw
    poids_quantile = poids_quantile_raw / total_raw

eps_depense = 0

if nb_req != nb_req_quantile:
    rho_utilise = rho_tot * (1 - poids_quantile)
    eps_depense = eps_from_rho_delta(rho_utilise, delta_tot)

    st.success("🔒 Budget pour le bruit gaussien")
    st.metric(label=r"Epsilon $\varepsilon$ dépensé (bruit gaussien)", value=f"{eps_depense:.4f}")
    st.metric(label=r"Delta $\delta$ utilisé", value=f"{delta_tot:.1e}")
    st.caption("Inclut les requêtes de type `count`, `sum` et `mean`")

if nb_req_quantile != 0:
    eps_rest = eps_tot - eps_depense

    st.warning("📊 Budget pour les quantiles")
    st.metric(label=r"Epsilon $\varepsilon$ alloué aux quantiles", value=f"{eps_rest:.4f}")
    st.caption("Réservé aux requêtes de type `quantile`")


# --- Sauvegarde dans la session pour accès inter-pages ---
st.session_state.poids_count = poids_count
st.session_state.poids_sum = poids_sum
st.session_state.poids_mean = poids_mean
st.session_state.poids_quantile = poids_quantile


st.session_state.eps_tot = eps_tot
st.session_state.delta_tot = delta_tot
st.session_state.rho_utilise = rho_utilise if nb_req != nb_req_quantile else 0
st.session_state.eps_rest = eps_rest if nb_req_quantile != 0 else 0


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


# --- Calcul (1 seule fois) ---
for i, (key, req) in enumerate(requetes.items(), 1):
    # Résultat DP
    resultat_dp = process_request_dp(context_rho, context_eps, key_values, req)
    st.markdown(f"### 🔐 Scale pour : `{key}`")
    scale = resultat_dp.summarize(alpha=0.05)["scale"]
    st.metric(label="DP scale (mean)", value=round(scale[0], 2))
