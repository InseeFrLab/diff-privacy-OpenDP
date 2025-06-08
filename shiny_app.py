# Imports
import json
from shiny import App, ui, render, reactive
import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.stats import norm
import seaborn as sns
from shinywidgets import render_widget
from app.initialisation import update_context
from src.process_tools import process_request_dp, process_request
from src.fonctions import eps_from_rho_delta, affichage_requete, affichage_requete_dp
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
    page_conception_budget,
    page_resultat_dp,
    page_etat_budget_dataset,
    page_a_propos
)

dp.enable_features("contrib")

data_example = sns.load_dataset("penguins")


def make_radio_buttons(request, filter_type: list[str]):
    radio_buttons = []
    priorite = {"Comptage": "2", "Total": "2", "Moyenne": "1", "Quantile": "3"}
    for key, req in request.items():
        if req["type"] in filter_type:
            radio_buttons_id = key
            radio_buttons.append(
                ui.input_radio_buttons(radio_buttons_id, key,
                    {"1": 1, "2": 2, "3": 3}, selected=priorite[req["type"]]
                )
            )
    return radio_buttons


def normalize_weights(request, input) -> dict:
    radio_to_weight = {1: 1, 2: 0.5, 3: 0.25}
    raw_weights = {
        key: radio_to_weight.get(float(getattr(input, key)()), 0)
        for key, requete in request.items()
    }
    total = sum(raw_weights.values())
    return {k: v / total for k, v in raw_weights.items()}


# 1. UI --------------------------------------
app_ui = ui.page_navbar(
    ui.nav_spacer(),
    page_donnees(),
    page_mecanisme_dp(),
    page_preparer_requetes(),
    page_conception_budget(),
    page_resultat_dp(),
    page_etat_budget_dataset(),
    page_a_propos(),
    title=ui.div(
        ui.img(src="insee-logo.jpg", height="80px", style="margin-right:10px"),
        ui.div("Poc – Differential Privacy", style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 400; font-size: 28px;"),
        style="display: flex; align-items: center; gap: 10px;"
    ),
    id="page",
)


# 2. Server ----------------------------------


def server(input, output, session):

    requetes = reactive.Value({})
    json_source = reactive.Value("")
    page_autorisee = reactive.Value(False)
    resultats_df = reactive.Value({})
    onglet_actuel = reactive.Value("Conception du budget")  # Onglet par défaut
    trigger_update_budget = reactive.Value(0)

    @reactive.calc
    def budgets_par_dataset():
        _ = trigger_update_budget()
        df = pd.read_csv("budget_dp.csv")

        # somme budget pour France entière par dataset
        df_france = df[df["echelle_geographique"] == "France entière"].groupby("dataset", as_index=False)["budget_dp_rho"].sum()
        df_france = df_france.rename(columns={"budget_dp_rho": "budget_france"})

        # somme budget pour chaque autre échelle par dataset
        df_autres = df[df["echelle_geographique"] != "France entière"].groupby(["dataset", "echelle_geographique"], as_index=False)["budget_dp_rho"].sum()

        # pour chaque dataset, on prend la valeur max des sommes sur les autres échelles
        df_max_autres = df_autres.groupby("dataset", as_index=False)["budget_dp_rho"].max()
        df_max_autres = df_max_autres.rename(columns={"budget_dp_rho": "budget_max_autres"})

        # merge budgets France entière et max autres échelles (outer pour ne rien perdre)
        df_merge = pd.merge(df_france, df_max_autres, on="dataset", how="outer").fillna(0)

        # somme finale
        df_merge["budget_dp_rho"] = df_merge["budget_france"] + df_merge["budget_max_autres"]

        # on trie et on ne garde que ce qui nous intéresse
        df_result = df_merge[["dataset", "budget_dp_rho"]].sort_values("budget_dp_rho", ascending=False)

        return df_result


    @output
    @render.ui
    def budget_display():
        df_grouped = budgets_par_dataset()

        boxes = []
        for _, row in df_grouped.iterrows():
            boxes.append(
                ui.value_box(
                    title=row["dataset"],
                    value=f"{row['budget_dp_rho']:.3f}"
                )
            )

        # Regrouper les value boxes en lignes de 4 colonnes max
        rows = []
        for i in range(0, len(boxes), 4):
            row = ui.row(*[ui.column(3, box) for box in boxes[i:i+4]])
            rows.append(row)

        return ui.div(*rows)


    @output
    @render.data_frame
    def data_budget_view():
        _ = trigger_update_budget()
        return pd.read_csv("budget_dp.csv")

    @reactive.Effect
    @reactive.event(input.confirm_validation)
    def _():
        page_autorisee.set(True)
        ui.modal_remove()
        ui.update_navs("page", selected="Résultat DP")

        nouvelle_ligne = pd.DataFrame([{
            "dataset": input.dataset_name(),
            "echelle_geographique": input.echelle_geo(),
            "date_ajout": datetime.now().strftime("%d/%m/%Y"),
            "budget_dp_rho": input.budget_total()
        }])

        fichier = Path("budget_dp.csv")
        if fichier.exists():
            nouvelle_ligne.to_csv(fichier, mode="a", header=False, index=False, encoding="utf-8")
        else:
            nouvelle_ligne.to_csv(fichier, mode="w", header=True, index=False, encoding="utf-8")

        ui.notification_show("✅ Ligne ajoutée à `budget_dp.csv`", type="message")
        trigger_update_budget.set(trigger_update_budget() + 1)  # 🔄 Déclenche la mise à jour

    # Stocker l'onglet actuel en réactif
    @reactive.Effect
    @reactive.event(input.page)
    def on_tab_change():
        requested_tab = input.page()
        if requested_tab == "Résultat DP" and not page_autorisee():
            # Afficher modal pour prévenir
            ui.modal_show(
                ui.modal(
                    "Vous devez valider le budget avant d'accéder aux résultats.",
                    title="Accès refusé",
                    easy_close=True,
                    footer=None
                )
            )
            # Remettre l'onglet actif sur l'onglet précédent (empêche le changement)
            ui.update_navs("page", selected=onglet_actuel())
        else:
            # Autoriser le changement d'onglet
            onglet_actuel.set(requested_tab)

    @reactive.Effect
    @reactive.event(input.valider_budget)
    def _():
        ui.modal_show(
            ui.modal(
                "Êtes-vous sûr de vouloir valider le budget ? Cette action est irréversible.",
                title="Confirmation",
                easy_close=False,
                footer=ui.TagList(
                    ui.input_action_button("confirm_validation", "Valider", class_="btn-danger"),
                    ui.input_action_button("cancel_validation", "Annuler", class_="btn-secondary")
                )
            )
        )

    @reactive.Effect
    @reactive.event(input.cancel_validation)
    def _():
        ui.modal_remove()

    @output
    @render.ui
    @reactive.event(input.confirm_validation)
    async def req_dp_display():
        data_requetes = requetes()
        weights = normalize_weights(requetes(), input)

        poids_requetes_rho = [
            weights[clef]
            for clef, requete in data_requetes.items()
            if requete["type"] != "Quantile"
        ]

        poids_requetes_quantile = [
            weights[clef]
            for clef, requete in data_requetes.items()
            if requete["type"] == "Quantile"
        ]

        context_param = {
            "data": pl.from_pandas(dataset()).lazy(),
            "privacy_unit": dp.unit_of(contributions=1),
            "margins": [dp.polars.Margin(max_partition_length=10000)],
        }

        context_rho, context_eps = update_context(
            context_param, input.budget_total(), poids_requetes_rho, poids_requetes_quantile
        )

        # --- Barre de progression ---
        with ui.Progress(min=0, max=len(data_requetes)) as p:
            p.set(0, message="Traitement en cours...", detail="Analyse requête par requête...")
            affichage = await affichage_requete_dp(context_rho, context_eps, key_values(), data_requetes, p, resultats_df)

        return affichage

    @output
    @render.download(filename=lambda: "resultats_dp.xlsx")
    def download_xlsx():
        import io
        import pandas as pd

        resultats = resultats_df()

        buffer = io.BytesIO()

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            for key, df in resultats.items():
                df.to_excel(writer, sheet_name=str(key)[:31], index=False)

        buffer.seek(0)
        return buffer

    @output
    @render.ui
    @reactive.event(input.request_input, input.add_req, input.delete_btn, input.delete_all_btn)
    def radio_buttons_comptage():
        return ui.layout_columns(
            *make_radio_buttons(requetes(), ["Comptage"]),
            col_widths=3
        )

    @output
    @render.ui
    @reactive.event(input.request_input, input.add_req, input.delete_btn, input.delete_all_btn)
    def radio_buttons_total_moyenne():
        return ui.layout_columns(
            *make_radio_buttons(requetes(), ["Total", "Moyenne"]),
            col_widths=3
        )

    @output
    @render.ui
    @reactive.event(input.request_input, input.add_req, input.delete_btn, input.delete_all_btn)
    def radio_buttons_quantile():
        return ui.layout_columns(
            *make_radio_buttons(requetes(), ["Quantile"]),
            col_widths=3
        )

    @render_widget
    def plot_comptage():
        df = pd.DataFrame(X_count())
        return create_barplot(df, x_col="requête", y_col="écart_type")

    @render_widget
    def plot_total_moyenne():
        df = pd.DataFrame(X_total_moyenne())
        if not df.empty:
            df = df.explode("cv (%)").reset_index(drop=True)
        return create_scatterplot(df, x_col="cv (%)", y_col="requête", size_col="cv (%)")

    @render_widget
    def plot_quantile():
        df = pd.DataFrame(X_quantile())
        return create_barplot(df, x_col="requête", y_col="candidats")

    @reactive.Calc
    def X_count():
        weights = normalize_weights(requetes(), input)
        results = []
        reqs = {k: v for k, v in requetes().items() if v["type"].lower() in ["count", "comptage"]}
        for i, (key, request) in enumerate(reqs.items()):
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

            resultat_dp = process_request_dp(context_rho, context_eps, key_values(), request)
            scale = resultat_dp.precision()["scale"][0]
            results.append({"requête": key, "écart_type": scale})

        return results

    
    @reactive.Calc
    def X_total_moyenne():
        weights = normalize_weights(requetes(), input)
        results = []
        reqs = {k: v for k, v in requetes().items() if v["type"].lower() in ["total", "moyenne"]}
        for i, (key, request) in enumerate(reqs.items()):
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

            resultat_dp = process_request_dp(context_rho, context_eps, key_values(), request)
            if request["type"] == "sum" or request["type"] == "Total":
                scale = resultat_dp.precision()["scale"]
                resultat = process_request(pl.from_pandas(dataset()).lazy(), request)
                list_cv = 100 * scale[0]/resultat["sum"]
                results.append({"requête": key, "cv (%)": list_cv})

            if request["type"] == "mean" or request["type"] == "Moyenne":
                scale_tot, scale_len = resultat_dp.precision()["scale"]
                resultat = process_request(pl.from_pandas(dataset()).lazy(), request)
                list_cv_tot = scale_tot/resultat["sum"]
                list_cv_len = scale_len/resultat["count"]
                list_cv = [100 * np.sqrt(list_cv_tot[i]**2 + list_cv_len[i]**2) for i in range(len(list_cv_tot))]
                results.append({"requête": key, "cv (%)": list_cv})

        return results

    @reactive.Calc
    def X_quantile():
        weights = normalize_weights(requetes(), input)
        results = []
        reqs = {k: v for k, v in requetes().items() if v["type"].lower() in ["quantile"]}
        for i, (key, request) in enumerate(reqs.items()):
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

            resultat_dp = process_request_dp(context_rho, context_eps, key_values(), request)
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

    @output
    @render.download(filename=lambda: "requetes_exportees.json")
    def download_json():
        import io
        import json

        buffer = io.StringIO()
        json.dump(requetes(), buffer, indent=2, ensure_ascii=False)
        buffer.seek(0)
        return buffer

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
