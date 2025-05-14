import streamlit as st
from app.initialisation import init_session_defaults, update_context
from fonctions import eps_from_rho_delta, rho_from_eps_delta
from process_tools import process_request_dp

init_session_defaults()

st.subheader("🔐 Paramètres de confidentialité")

col1, col2, col3 = st.columns(3)
eps_tot = col1.slider(r"Epsilon $\varepsilon$", 0.1, 10.0, 3.0, step=0.1)
delta_exp = col2.slider(r"Delta $\delta = 10^{x}$", -10, -1, -5)
delta_tot = 10 ** (delta_exp)

# --- Calcul et affichage de rho ---
rho_tot = rho_from_eps_delta(eps_tot, delta_tot)
col3.metric(label=r"Budget équivalent en Rho $\rho$", value=f"{rho_tot:.4f}")

st.markdown("### 🔧 Répartition du budget ρ par requête")
requetes = st.session_state.requetes
nb_req = len(requetes)
poids_raw = {}
clefs_requetes = list(requetes.keys())

# On divise en groupes de 3 sliders par ligne
for i in range(0, nb_req, 3):
    cols = st.columns(3)
    for j in range(3):
        idx = i + j
        if idx < nb_req:
            key = clefs_requetes[idx]
            req = requetes[key]
            label = f"{key} – {req['type']}"
            slider_value = cols[j].slider(
                label,
                min_value=0.0,
                max_value=1.0,
                value=1.0 / nb_req,
                step=0.01,
                key=f"slider_{key}"
            )
            poids_raw[key] = slider_value

# --- Normalisation automatique ---
total_raw = sum(poids_raw.values())

if total_raw == 0:
    st.warning("⚠️ La somme des poids est nulle. Les poids ne peuvent pas être normalisés.")
    poids_normalises = {i: 0.0 for i in poids_raw}
else:
    poids_normalises = {i: v / total_raw for i, v in poids_raw.items()}

eps_depense = 0

somme_poids_quantile = sum(
    poids_normalises[clef]
    for clef, req in requetes.items()
    if req["type"] == "quantile"
)

df = st.session_state.df
context_param = st.session_state.context_param
key_values = st.session_state.key_values

# Liste des poids, dans l’ordre des requêtes (hors quantile)
poids_requetes_rho = [
    poids_normalises[clef]
    for clef, req in requetes.items()
    if req["type"] != "quantile"
]

poids_requetes_quantile = [
    poids_normalises[clef]
    for clef, req in requetes.items()
    if req["type"] == "quantile"
]

st.session_state.poids_requetes_rho = poids_requetes_rho
st.session_state.poids_requetes_quantile = poids_requetes_quantile

# Initialisation de la barre dans la sidebar
progress_bar = st.sidebar.progress(0, text="Progression du traitement des requêtes...")

# --- Calcul (1 seule fois) ---
cols_per_row = 3
keys = list(requetes.keys())
nb_req = len(keys)
(context_rho, context_eps) = update_context(eps_tot, delta_tot, poids_requetes_rho, poids_requetes_quantile)

for i in range(0, nb_req, cols_per_row):
    cols = st.columns(cols_per_row)

    for j in range(cols_per_row):
        idx = i + j
        if idx < nb_req:
            key = keys[idx]
            req = requetes[key]

            with cols[j]:
                # Résultat DP
                resultat_dp = process_request_dp(context_rho, context_eps, key_values, req)
                scale = resultat_dp.summarize(alpha=0.05)["scale"]
                resultat_dp.release()
                st.markdown(f"#### 🔐 `{key}`")
                st.metric(label="DP scale", value=round(scale[0], 2))

                st.caption(f"Requête {idx+1} sur {nb_req}")

            # Barre de progression en bas de chaque ligne
            progress_bar.progress(int(100 * (idx+1) / nb_req), text=f"Progression : {idx+1}/{nb_req}")

if nb_req != st.session_state.nb_req_quantile:
    rho_utilise = rho_tot * (1 - somme_poids_quantile)
    eps_depense = eps_from_rho_delta(rho_utilise, delta_tot)

    st.success("🔒 Budget pour le bruit gaussien")
    st.metric(label=r"Epsilon $\varepsilon$ dépensé (bruit gaussien)", value=f"{eps_depense:.4f}")
    st.metric(label=r"Delta $\delta$ utilisé", value=f"{delta_tot:.1e}")
    st.caption("Inclut les requêtes de type `count`, `sum` et `mean`")

if st.session_state.nb_req_quantile != 0:
    eps_rest = eps_tot - eps_depense

    st.warning("📊 Budget pour les quantiles")
    st.metric(label=r"Epsilon $\varepsilon$ alloué aux quantiles", value=f"{eps_rest:.4f}")
    st.caption("Réservé aux requêtes de type `quantile`")


# --- Sauvegarde dans la session pour accès inter-pages ---
st.session_state.eps_tot = eps_tot
st.session_state.delta_tot = delta_tot
