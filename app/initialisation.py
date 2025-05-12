import streamlit as st
import json
import os
import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fonctions import eps_from_rho_delta, rho_from_eps_delta


def init_session_defaults():

    # Chargement des données par défaut (si non présent)
    if "data_json" not in st.session_state:
        try:
            with open("data_example.json", "r", encoding="utf-8") as f:
                st.session_state.data_json = json.load(f)
                st.session_state.source = "📦 Fichier par défaut : `data_example.json`"
        except FileNotFoundError:
            st.session_state.data_json = {}
            st.session_state.source = None
            st.error("❌ Fichier par défaut introuvable.")

    # Poids normalisés par défaut
    st.session_state.setdefault("poids_count", 0.25)
    st.session_state.setdefault("poids_sum", 0.25)
    st.session_state.setdefault("poids_mean", 0.25)
    st.session_state.setdefault("poids_quantile", 0.25)

    # Paramètres de confidentialité par défaut
    st.session_state.setdefault("eps_tot", 3.0)
    st.session_state.setdefault("delta_tot", 1e-5)
