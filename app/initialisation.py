import streamlit as st
import json
from app.constants import DF_POLARS, CONTEXT_PARAM, KEY_VALUES
from fonctions import eps_from_rho_delta, rho_from_eps_delta
import opendp.prelude as dp

dp.enable_features("contrib")


def init_session_defaults():

    # Lecture du fichier JSON par défaut (seulement si non présent)
    if "requetes" not in st.session_state:
        try:
            with open("data_example.json", "r", encoding="utf-8") as f:
                st.session_state.requetes = json.load(f)
                st.session_state.source = "📦 Fichier par défaut : `data_example.json`"
                update_info_request()

        except FileNotFoundError:
            st.session_state.requetes = {}
            st.session_state.source = None
            st.session_state.setdefault("nb_req_count", 0)
            st.session_state.setdefault("nb_req_sum", 0)
            st.session_state.setdefault("nb_req_mean", 0)
            st.session_state.setdefault("nb_req_quantile", 0)
            st.session_state.setdefault("poids_requetes_rho", 0)
            st.session_state.setdefault("poids_requetes_quantile", 0)
            st.error("❌ Fichier par défaut introuvable.")

    # Paramètres généraux sur le dataset
    st.session_state.setdefault("df", DF_POLARS)
    st.session_state.setdefault("context_param", CONTEXT_PARAM)
    st.session_state.setdefault("key_values", KEY_VALUES)

    # Confidentialité différentielle
    st.session_state.setdefault("eps_tot", 3.0)
    st.session_state.setdefault("delta_tot", 1e-5)


def update_info_request():
    # --- Comptage des types de requêtes ---
    requetes = st.session_state.requetes

    st.session_state.nb_req_count = sum(1 for req in requetes.values() if req.get("type") == "count")
    st.session_state.nb_req_sum = sum(1 for req in requetes.values() if req.get("type") == "sum")
    st.session_state.nb_req_mean = sum(1 for req in requetes.values() if req.get("type") == "mean")
    st.session_state.nb_req_quantile = sum(1 for req in requetes.values() if req.get("type") == "quantile")

    poids_requetes_rho = [
        1 / len(requetes)
        for req in requetes.values()
        if req.get("type") != "quantile"
    ]

    poids_requetes_quantile = [
        1 / len(requetes)
        for req in requetes.values()
        if req.get("type_req") == "quantile"
    ]

    st.session_state.poids_requetes_rho = poids_requetes_rho
    st.session_state.poids_requetes_quantile = poids_requetes_quantile


def update_context(eps_tot, delta_tot, poids_requetes_rho, poids_requetes_quantile):

    rho_utilise = rho_from_eps_delta(eps_tot, delta_tot) * (1 - sum(poids_requetes_quantile))
    eps_rest = eps_tot - eps_from_rho_delta(rho_utilise, delta_tot)

    context_rho = dp.Context.compositor(
            **CONTEXT_PARAM,
            privacy_loss=dp.loss_of(rho=rho_utilise),
            split_by_weights=poids_requetes_rho
        )

    context_eps = dp.Context.compositor(
            **CONTEXT_PARAM,
            privacy_loss=dp.loss_of(epsilon=eps_rest),
            split_by_weights=poids_requetes_quantile
        )

    return context_rho, context_eps
