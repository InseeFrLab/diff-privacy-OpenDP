import streamlit as st
import numpy as np
from app.initialisation import init_session_defaults, update_context
from src.fonctions import eps_from_rho_delta, format_scientifique, add_bar, add_scatter
from src.process_tools import process_request_dp, process_request
from plotly.subplots import make_subplots
import pandas as pd

init_session_defaults()


def format_scale(scale):
    # Si c'est un scalaire
    if np.isscalar(scale):
        return f"{scale:.1f}"

    # Sinon, liste de valeurs
    return "(" + ", ".join(f"{s:.1f}" if abs(s) < 1000 else f"{s:.1e}" for s in scale) + ")"


tab1, tab2 = st.tabs(["Budget", "Rappel requêtes"])

with tab1:
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
                    min_value=0.01,
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

    resultats_count = []
    resultats_sum = []
    resultats_mean = []
    resultats_quantile = []

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

        (context_rho, context_eps) = update_context(st.session_state.context_param, rho_budget, poids_requetes_rho, poids_requetes_quantile)

        resultat_dp = process_request_dp(context_rho, context_eps, key_values, req)

        if req_type == "count":
            scale = resultat_dp.precision()["scale"][0]
            resultats_count.append({"requête": key, "écart_type": scale})

        elif req_type == "sum":
            scale = resultat_dp.precision()["scale"]
            resultat = process_request(st.session_state.df.lazy(), req)
            list_cv = 100 * scale[0]/resultat["sum"]
            resultats_sum.append({"requête": key, "cv (%)": list_cv})

        elif req_type == "mean":
            scale_tot, scale_len = resultat_dp.precision()["scale"]
            resultat = process_request(st.session_state.df.lazy(), req)
            list_cv_tot = scale_tot/resultat["sum"]
            list_cv_len = scale_len/resultat["count"]
            list_cv = [100 * np.sqrt(list_cv_tot[i]**2 + list_cv_len[i]**2) for i in range(len(list_cv_tot))]
            resultats_mean.append({"requête": key, "cv (%)": list_cv, "cv_tot (%)": 100 * list_cv_tot, "cv_len (%)": 100 * list_cv_len})

        else:  # quantile
            nb_candidat = resultat_dp.precision(
                data=st.session_state.df.lazy(),
                epsilon=np.sqrt(8 * rho_budget * sum(poids_requetes_quantile)) * poids_requetes_quantile[0]
            )
            resultats_quantile.append({"requête": key, "candidats": nb_candidat})

        processed += 1
        progress_bar.progress(int(100 * processed / total), text=f"Progression : {processed}/{total}")

    df_count = pd.DataFrame(resultats_count)

    df = pd.DataFrame(resultats_mean)
    df_mean = df.explode(["cv (%)", "cv_tot (%)", "cv_len (%)"]).reset_index(drop=True)

    df = pd.DataFrame(resultats_sum)
    df_sum = df.explode("cv (%)").reset_index(drop=True)

    df_quantile = pd.DataFrame(resultats_quantile)

    # Création de la figure en 2x2
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Écart type du bruit (Count)",
            "Nombre de candidats (Quantile)",
            "Coefficient de variation (%) (Sum)",
            "Coefficient de variation (%) (Mean)"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # Ajout des barplots dans la figure
    add_bar(fig, df_count, row=1, col=1, x_col="requête", y_col="écart_type", color="#636EFA")
    add_bar(fig, df_quantile, row=1, col=2, x_col="requête", y_col="candidats", color="#AB63FA")

    add_scatter(fig, df_sum, row=2, col=1, x_col="cv (%)", y_col="requête", size_col="cv (%)")
    add_scatter(fig, df_mean, row=2, col=2, x_col="cv_tot (%)", y_col="cv_len (%)", size_col="cv (%)")

    fig.update_yaxes(title_text="Écart type", row=1, col=1)

    fig.update_yaxes(title_text="Nombre de candidats", row=1, col=2)

    fig.update_xaxes(title_text="CV (%)", row=2, col=1)

    fig.update_xaxes(title_text="CV total (%)", row=2, col=2)
    fig.update_yaxes(title_text="CV comptage (%)", row=2, col=2)

    # Mise en forme générale
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text="",
        title_font_size=20,
        margin=dict(t=60, b=40, l=40, r=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14)
    )

    # Affichage dans Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # --- Sauvegarde dans la session pour accès inter-pages ---
    st.session_state.rho_budget = rho_budget

with tab2:
    for key, req in requetes.items():
        with st.expander(f"🔎 {key} — {req['type'].upper()}", expanded=True):
            col1, col2 = st.columns(2)

            # Colonne 1 : infos principales
            with col1:
                st.markdown(f"**Variable :** `{req.get('variable', '—')}`")
                if req.get("alpha") is not None:
                    st.markdown(f"**Alpha :** `{req['alpha']}`")
                bounds = req.get("bounds")
                if bounds:
                    st.markdown(f"**Bornes :** min=`{bounds[0]}`, max=`{bounds[1]}`")
                else:
                    st.markdown("**Bornes :** `—`")

            # Colonne 2 : conditions
            with col2:
                st.markdown(f"**Filtre :** `{req.get('filtre') or '—'}`")
                by = req.get("by")
                st.markdown(f"**Par :** `{', '.join(by) if by else '—'}`")

                if req.get("candidats") is not None:
                    cands = req["candidats"]
                    if cands["type"] == "list":
                        vals = ", ".join(map(str, cands["values"]))
                        st.markdown(f"**Candidats (liste) :** `{vals}`")
                    else:
                        st.markdown(f"**Candidats (range) :** min=`{cands['min']}`, max=`{cands['max']}`, step=`{cands['step']}`")