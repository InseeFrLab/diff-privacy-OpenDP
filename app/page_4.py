import streamlit as st
import numpy as np
from app.initialisation import init_session_defaults, update_context
from fonctions import eps_from_rho_delta
from process_tools import process_request_dp

init_session_defaults()


def format_scientifique(val, precision=2):
    base, exposant = f"{val:.{precision}e}".split("e")
    exposant = int(exposant)  # conversion propre de l'exposant
    return f"10^{exposant}"


def format_scale(scale):
    # Si c'est un scalaire
    if np.isscalar(scale):
        return f"{scale:.1f}" if abs(scale) < 1000 else f"{scale:.1e}"

    # Sinon, liste de valeurs
    return "(" + ", ".join(f"{s:.1f}" if abs(s) < 1000 else f"{s:.1e}" for s in scale) + ")"


# Titre
st.subheader("🔐 Paramètres de confidentialité")
with st.container():
    col1, col2, col3, col4 = st.columns(4)

    rho_depense_init = col1.slider(r"Budget $\rho$ déjà dépensé", 0.0, 1.0, value=0.1, step=0.01)
    rho_budget = col2.slider(r"Budget $\rho$ de l'étude", 0.01, 1.0, value=0.1)

    delta_exp = st.sidebar.slider(r"Delta $\delta = 10^{x}$", -10, -1, -5)
    delta = 10 ** (delta_exp)
    epsilon = eps_from_rho_delta(rho_depense_init + rho_budget, delta)

    col3.metric("Budget total en ρ", f"{rho_depense_init + rho_budget:.4f}")
    col4.metric("Budget total en (ε, δ)", f"({epsilon:.4f}, {format_scientifique(delta)})")
# Marqueur de fin
st.markdown("---")

st.markdown("### 🔧 Répartition du budget ρ par requête")
requetes = st.session_state.requetes
nb_req = len(requetes)
poids_raw = {}
clefs_requetes = list(requetes.keys())

# --- Organisation en colonnes verticales par type de requête ---
types_possibles = ["count", "sum", "mean", "quantile"]
type_to_keys = {t: [k for k, v in requetes.items() if v["type"] == t] for t in types_possibles}

# Création des colonnes : une par type
cols = st.columns(len(types_possibles))

for col, req_type in zip(cols, types_possibles):
    with col:
        st.markdown(f"#### `{req_type}`")
        for key in type_to_keys[req_type]:
            req = requetes[key]
            slider_value = st.slider(
                f"**{key}**",
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

# --- Organisation des résultats en colonnes par type ---
st.markdown("---")
st.markdown("### 📊 Résultats des requêtes (DP) par type")

type_to_col = {t: col for t, col in zip(types_possibles, st.columns(len(types_possibles)))}

processed = 0
total = len(keys)

count_rho = -1
count_quantile = -1

# Affichage dans l'ordre des requêtes, mais dans la colonne du type
for key in keys:
    req = requetes[key]
    req_type = req["type"]
    col = type_to_col[req_type]

    if req_type != "quantile":
        count_rho += 1
        poids_requetes_rho[0], poids_requetes_rho[count_rho] = poids_requetes_rho[count_rho], poids_requetes_rho[0]
    else:
        count_quantile += 1
        poids_requetes_quantile[0], poids_requetes_quantile[count_quantile] = poids_requetes_quantile[count_quantile], poids_requetes_quantile[0]

    (context_rho, context_eps) = update_context(rho_budget, poids_requetes_rho, poids_requetes_quantile)

    with col:
        st.markdown(f"##### 🔐 `{key}`")
        resultat_dp = process_request_dp(context_rho, context_eps, key_values, req)
        print(1)
        scale = resultat_dp.summarize(alpha=0.05)["scale"]
        st.metric(label="Ecart type du bruit injecté", value=format_scale(scale))
        st.caption(f"Requête {processed + 1} sur {total}")

    processed += 1
    progress_bar.progress(int(100 * processed / total), text=f"Progression : {processed}/{total}")

# --- Sauvegarde dans la session pour accès inter-pages ---
st.session_state.rho_budget = rho_budget
