# Imports
import json
from shiny import App, ui, render, reactive
import pandas as pd
import polars as pl
from pathlib import Path
import numpy as np
from scipy.stats import norm
import seaborn as sns
from shinywidgets import render_widget
from app.initialisation import update_context
from src.process_tools import process_request_dp, process_request
from src.fonctions import eps_from_rho_delta, affichage_requete
import opendp.prelude as dp
from plots import (
    create_histo_plot, create_fc_emp_plot,
    create_score_plot, create_proba_plot,
    create_barplot, create_scatterplot
)
from layout import (
    page_donnees,
    page_preparer_requetes,
    page_mecanisme_dp,
    page_conception_budget
)

dp.enable_features("contrib")

# --- Etat de session réactif ---
requetes = reactive.Value({})  # Dictionnaire des requêtes
json_source = reactive.Value("")

type_req = ["Comptage", "Total", "Moyenne", "Quantile"]
data_example = sns.load_dataset("penguins")


def make_radio_buttons(filter_type: list[str]):
    radio_buttons = []
    priorite = {"Comptage": "2", "Total": "2", "Moyenne": "1", "Quantile": "3"}
    for key, req in requetes().items():
        if req["type"] in filter_type:
            radio_buttons_id = key
            radio_buttons.append(
                ui.input_radio_buttons(radio_buttons_id, key,
                    {"1": 1, "2": 2, "3": 3}, selected=priorite[req["type"]]
                )
            )
    return radio_buttons


def normalize_weights(input) -> dict:
    radio_to_weight = {1: 1, 2: 0.5, 3: 0.25}
    reqs = requetes()

    raw_weights = {
        key: radio_to_weight.get(float(getattr(input, key)()), 0)
        for key, req in reqs.items()
    }

    total = sum(raw_weights.values())
    return {k: v / total for k, v in raw_weights.items()}


# 1. UI --------------------------------------
app_ui = ui.page_navbar(
    ui.nav_spacer(),
    page_donnees(),
    page_preparer_requetes(),
    page_mecanisme_dp(),
    page_conception_budget(),
    ui.nav_panel("Résultat DP", ui.h3("À compléter...")),
    ui.nav_panel("Etat budget dataset", ui.h3("À compléter...")),
    ui.nav_panel("A propos", ui.h3("À compléter...")),
    title=ui.div(
        ui.img(src="insee-logo.jpg", height="80px", style="margin-right:10px"),
        ui.div("Poc – Differential Privacy", style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 400; font-size: 28px;"),
        style="display: flex; align-items: center; gap: 10px;"
    ),
    id="page"
)


# 2. Server ----------------------------------


def server(input, output, session):

    @output
    @render.ui
    @reactive.event(input.request_input, input.add_req, input.delete_btn, input.delete_all_btn)
    def radio_buttons_comptage():
        return ui.layout_columns(
            *make_radio_buttons(["Comptage"]),
            col_widths=3
        )

    @output
    @render.ui
    @reactive.event(input.request_input, input.add_req, input.delete_btn, input.delete_all_btn)
    def radio_buttons_total_moyenne():
        return ui.layout_columns(
            *make_radio_buttons(["Total", "Moyenne"]),
            col_widths=3
        )

    @output
    @render.ui
    @reactive.event(input.request_input, input.add_req, input.delete_btn, input.delete_all_btn)
    def radio_buttons_quantile():
        return ui.layout_columns(
            *make_radio_buttons(["Quantile"]),
            col_widths=3
        )

    @render_widget
    def plot_comptage():
        df = pd.DataFrame(X_count())
        return create_barplot(df, x_col="requête", y_col="écart_type")


    @render_widget
    def plot_total_moyenne():
        df = pd.DataFrame(X_total_moyenne())
        df = df.explode("cv (%)").reset_index(drop=True)

        return create_scatterplot(df, x_col="cv (%)", y_col="requête", size_col="cv (%)")

    @render_widget
    def plot_quantile():
        df = pd.DataFrame(X_quantile())
        return create_barplot(df, x_col="requête", y_col="candidats")

    
    @reactive.Calc
    def X_count():
        weights = normalize_weights(input)
        results = []
        reqs = {k: v for k, v in requetes().items() if v["type"].lower() in ["count", "comptage"]}

        for i, (key, req) in enumerate(reqs.items()):
            poids = list(weights.values())
            if poids:
                # mettre le poids courant en premier
                poids[0], poids[i] = poids[i], poids[0]

            context_param = {
                "data": pl.from_pandas(dataset()).lazy(),
                "privacy_unit": dp.unit_of(contributions=1),
                "margins": [dp.polars.Margin(max_partition_length=10000)],
            }

            context_rho, context_eps = update_context(context_param, input.budget_total(), poids, [])

            resultat_dp = process_request_dp(context_rho, context_eps, key_values(), req)
            scale = resultat_dp.precision()["scale"][0]
            results.append({"requête": key, "écart_type": scale})

        return results

    
    @reactive.Calc
    def X_total_moyenne():
        weights = normalize_weights(input)
        results = []
        reqs = {k: v for k, v in requetes().items() if v["type"].lower() in ["total", "moyenne"]}

        for i, (key, req) in enumerate(reqs.items()):
            poids = list(weights.values())
            if poids:
                # mettre le poids courant en premier
                poids[0], poids[i] = poids[i], poids[0]

            context_param = {
                "data": pl.from_pandas(dataset()).lazy(),
                "privacy_unit": dp.unit_of(contributions=1),
                "margins": [dp.polars.Margin(max_partition_length=10000)],
            }

            context_rho, context_eps = update_context(context_param, input.budget_total(), poids, [])

            resultat_dp = process_request_dp(context_rho, context_eps, key_values(), req)
            if req["type"] == "sum" or req["type"] == "Total":
                scale = resultat_dp.precision()["scale"]
                resultat = process_request(pl.from_pandas(dataset()).lazy(), req)
                list_cv = 100 * scale[0]/resultat["sum"]
                results.append({"requête": key, "cv (%)": list_cv})

            if req["type"] == "mean" or req["type"] == "Moyenne":
                scale_tot, scale_len = resultat_dp.precision()["scale"]
                resultat = process_request(pl.from_pandas(dataset()).lazy(), req)
                list_cv_tot = scale_tot/resultat["sum"]
                list_cv_len = scale_len/resultat["count"]
                list_cv = [100 * np.sqrt(list_cv_tot[i]**2 + list_cv_len[i]**2) for i in range(len(list_cv_tot))]
                results.append({"requête": key, "cv (%)": list_cv})

        return results

    @reactive.Calc
    def X_quantile():
        weights = normalize_weights(input)
        results = []
        reqs = {k: v for k, v in requetes().items() if v["type"].lower() in ["quantile"]}

        for i, (key, req) in enumerate(reqs.items()):
            poids = list(weights.values())
            if poids:
                # mettre le poids courant en premier
                poids[0], poids[i] = poids[i], poids[0]

            context_param = {
                "data": pl.from_pandas(dataset()).lazy(),
                "privacy_unit": dp.unit_of(contributions=1),
                "margins": [dp.polars.Margin(max_partition_length=10000)],
            }

            context_rho, context_eps = update_context(context_param, input.budget_total(), [], poids)

            resultat_dp = process_request_dp(context_rho, context_eps, key_values(), req)
            nb_candidat = resultat_dp.precision(
                data=pl.from_pandas(dataset()).lazy(),
                epsilon=np.sqrt(8 * input.budget_total() * sum(poids)) * poids[0]
            )
            results.append({"requête": key, "candidats": nb_candidat})

        return results


    # Page 1 ----------------------------------

    # Lire le dataset
    @reactive.Calc
    def dataset():
        file = input.dataset_input()
        if file is not None:
            ext = Path(file["name"]).suffix
            if ext == ".csv":
                return pd.read_csv(file["datapath"])
            elif ext == ".parquet":
                return pd.read_parquet(file["datapath"])
            else:
                raise ValueError("Format non supporté : utiliser CSV ou Parquet")
        else:
            return sns.load_dataset(input.default_dataset())

    # Afficher le dataset
    @output
    @render.data_frame
    def data_view():
        return dataset()

    # Afficher le résumé statistique
    @output
    @render.data_frame
    def data_summary():
        df = dataset().describe(include="all")
        df = df.round(2)
        df.insert(0, "Statistique", df.index)  # ajouter la colonne "Statistique"
        df.reset_index(drop=True, inplace=True)
        return df

    # Page 2 ----------------------------------

    @reactive.Calc
    def variable_choices():
        df = dataset()
        if df is None:
            return {}

        qualitative = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        quantitative = df.select_dtypes(include=["number"]).columns.tolist()
        return {
            "": "",
            "🔤 Qualitatives": {col: col for col in qualitative},
            "🧮 Quantitatives": {col: col for col in quantitative}
        }

    @reactive.Calc
    def key_values():
        df = dataset()
        qualitatif_cols = df.select_dtypes(include=["object", "category"]).columns
        return {
            col: sorted(df[col].dropna().unique().tolist())
            for col in qualitatif_cols
        }

    @reactive.Effect
    def update_variable_choices():
        # Met à jour dynamiquement les choix de la selectize input
        ui.update_selectize("variable", choices=variable_choices())
        ui.update_selectize("group_by", choices=variable_choices())

    @reactive.effect
    @reactive.event(input.request_input)
    def _():
        fileinfo = input.request_input()
        if fileinfo is not None:
            filepath = Path(fileinfo[0]["datapath"])
            try:
                with filepath.open(encoding="utf-8") as f:
                    data = json.load(f)
                requetes.set(data)
                json_source.set(filepath.name)
            except json.JSONDecodeError:
                ui.notification_show("❌ Fichier JSON invalide", type="error")

    @reactive.effect
    @reactive.event(input.save_json)
    def _():
        save_path = "requetes_exportées.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(requetes(), f, indent=2, ensure_ascii=False)
        ui.notification_show(f"✅ Requêtes exportées dans `{save_path}`", type="message")

    @reactive.effect
    @reactive.event(input.add_req)
    def _():
        current = requetes()
        i = 1
        while f"req_{i}" in current:
            i += 1
        new_id = f"req_{i}"

        raw_min = input.borne_min()
        raw_max = input.borne_max()

        bounds = [float(raw_min), float(raw_max)] if raw_min != "" and raw_max != "" else None

        base_dict = {
            "type": input.type_req(),
            "variable": input.variable(),
            "bounds": bounds,
            "by": input.group_by(),
            "filtre": input.filtre(),
        }

        if input.type_req() == 'Quantile':
            base_dict.update({
                "alpha": float(input.alpha()),
                "candidat": input.candidat(),
            })

        current[new_id] = {
            k: v for k, v in base_dict.items()
            if v not in [None, "", (), ["", ""]]
        }
        requetes.set(current)
        ui.notification_show(f"✅ Requête `{new_id}` ajoutée", type="message")
        ui.update_selectize("delete_req", choices=list(requetes().keys()))

    @reactive.effect
    @reactive.event(input.delete_btn)
    def _():
        current = requetes()
        targets = input.delete_req()  # ceci est une liste ou un tuple de valeurs

        if not targets:
            ui.notification_show("❌ Aucune requête sélectionnée", type="error")
            return

        removed = []
        not_found = []
        for target in targets:
            if target in current:
                del current[target]
                removed.append(target)
            else:
                not_found.append(target)

        requetes.set(current)

        if removed:
            ui.notification_show(f"🗑️ Requête(s) supprimée(s) : {', '.join(removed)}", type="warning")
            ui.update_selectize("delete_req", choices=list(requetes().keys()))
        if not_found:
            ui.notification_show(f"❌ Requête(s) introuvable(s) : {', '.join(not_found)}", type="error")


    @reactive.effect
    @reactive.event(input.delete_all_btn)
    def _():
        current = requetes()
        if current:
            current.clear()  # Vide toutes les requêtes
            requetes.set(current)  # Met à jour le reactive.Value
            ui.notification_show(f"🗑️ TOUTES les requêtes ont été supprimé", type="warning")   
            ui.update_selectize("delete_req", choices=list(requetes().keys()))

    @output
    @render.ui
    @reactive.event(input.request_input, input.add_req, input.delete_btn, input.delete_all_btn)
    def req_display():
        data_requetes = requetes()
        if not data_requetes:
            return ui.p("Aucune requête chargée.")
        return affichage_requete(data_requetes, dataset())

    # Page 3 ----------------------------------

    @render.ui
    def interval_summary():
        sigma = input.scale_gauss()
        quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
        result_lines = []
        for q in quantiles:
            z = norm.ppf(0.5 + q / 2)
            bound = round(z * sigma, 3)
            result_lines.append(f"<li><strong>{int(q * 100)}%</strong> de chances que le bruit soit entre +/- <code>{round(bound,1)}</code></li>")

        return ui.HTML("""
            <div style='margin-top:20px; padding:10px; background-color:#f9f9f9; border-radius:12px; font-family: "Raleway", "Garamond", sans-serif; font-size:16px; color:#333'>
                <p style="margin-bottom:10px"><strong>Résumé des intervalles de confiance :</strong></p>
                <ul style="padding-left: 20px; margin: 0;">
                    {}
                </ul>
            </div>
        """.format("".join(result_lines)))

    @output
    @render.data_frame
    def cross_table():
        table = data_example.groupby(["species", "island"]).size().unstack(fill_value=0)
        flat_table = table.reset_index().melt(id_vars="species", var_name="island", value_name="count")
        return flat_table.sort_values(by=["species", "island"])

    @output
    @render.data_frame
    @reactive.event(input.scale_gauss)
    def cross_table_2():
        from src.request_class import count_dp

        key_values = {
            "species": ["Adelie", "Chinstrap", "Gentoo"],
            "island": ["Biscoe", "Dream", "Torgersen"]
        }

        df_lazy = pl.from_pandas(data_example).lazy()

        context_rho = dp.Context.compositor(
            data=df_lazy,
            privacy_unit=dp.unit_of(contributions=1),
            privacy_loss=dp.loss_of(rho=1/(2*input.scale_gauss()**2)),
            split_by_weights=[1],
            margins=[
                dp.polars.Margin(max_partition_length=1000)
            ]
        )

        tableau_dp = count_dp(context_rho, key_values, by=["species", "island"], variable=None, filtre=None).execute().release().collect()

        tableau_dp = tableau_dp.sort(by=["species", "island"])
        first_col = tableau_dp.columns[0]
        new_order = tableau_dp.columns[1:] + [first_col]
        tableau_dp = tableau_dp.select(new_order)

        return tableau_dp

    @output
    @render.ui
    def dp_budget_summary():
        rho = 1 / (2 * input.scale_gauss() ** 2)
        delta_exp = input.delta_slider()
        delta = f"1e{delta_exp}"
        eps = eps_from_rho_delta(rho, 10**delta_exp)

        return ui.HTML(f"""
            <div style='margin-top:20px; padding:10px; background-color:#f9f9f9; border-radius:12px;
                        font-family: "Raleway", "Garamond", sans-serif; font-size:16px; color:#333'>
                <p style="margin-bottom:10px"><strong>Budget de confidentialité différentielle :</strong></p>
                <ul style="padding-left: 20px; margin: 0;">
                    <li>En zCDP, <strong>ρ</strong> = <code>{rho:.4f}</code></li>
                    <li>En Approximate DP, (<strong>ε</strong> = <code>{eps:.3f}</code>, <strong>δ</strong> = <code>{delta}</code>)</li>
                </ul>
            </div>
        """)

    @render.plot
    def histo_plot():
        return create_histo_plot(data_example, input.alpha_slider())

    @render.plot
    def fc_emp_plot():
        return create_fc_emp_plot(data_example, input.alpha_slider())

    @render_widget
    def score_plot():
        candidat_min, candidat_max = input.candidat_slider()
        return create_score_plot(
            data_example, input.alpha_slider(), input.epsilon_slider(),
            candidat_min, candidat_max, input.candidat_step()
        )

    @render_widget
    def proba_plot():
        candidat_min, candidat_max = input.candidat_slider()
        return create_proba_plot(
            data_example, input.alpha_slider(), input.epsilon_slider(),
            candidat_min, candidat_max, input.candidat_step()
        )


www_dir = Path(__file__).parent / "www"

app = App(app_ui, server, static_assets=www_dir)
# shiny run --reload shiny_app.py
