# Imports
import json
from shiny import App, ui, render, reactive
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from shinywidgets import output_widget, render_widget

# --- Etat de session réactif ---
requetes = reactive.Value({})  # Dictionnaire des requêtes
json_source = reactive.Value("")

type_req = ["Comptage", "Total", "Moyenne", "Quantile"]

def manual_quantile_score(data, candidats, alpha, et_si=False):
    if alpha == 0:
        alpha_num, alpha_denum = 0, 1
    elif alpha == 0.25:
        alpha_num, alpha_denum = 1, 4
    elif alpha == 0.5:
        alpha_num, alpha_denum = 1, 2
    elif alpha == 0.75:
        alpha_num, alpha_denum = 3, 4
    elif alpha == 1:
        alpha_num, alpha_denum = 1, 1
    else:
        alpha_num = int(np.floor(alpha * 10_000))
        alpha_denum = 10_000

    if et_si == True:
        alpha_num = int(np.floor(alpha * 10_000))
        alpha_denum = 10_000

    scores = []
    for c in candidats:
        n_less = np.sum(data < c)
        n_equal = np.sum(data == c)
        score = alpha_denum * n_less - alpha_num * (len(data) - n_equal)
        scores.append(abs(score))

    return np.array(scores), max(alpha_num, alpha_denum - alpha_num)


def make_card_body(req):
    parts = []

    fields = [
        ("variable", "📌 Variable", lambda v: f"`{v}`"),
        ("bounds", "🎯 Bornes", lambda v: f"`[{v[0]}, {v[1]}]`" if isinstance(v, list) and len(v) == 2 else "—"),
        ("by", "🧷 Group by", lambda v: f"`{', '.join(v)}`"),
        ("filtre", "🧮 Filtre", lambda v: f"`{v}`"),
        ("alpha", "📈 Alpha", lambda v: f"`{v}`"),
    ]

    for key, label, formatter in fields:
        val = req.get(key)
        if val is not None and val != "" and val != []:
            parts.append(ui.p(f"{label} : {formatter(val)}"))

    return ui.card_body(*parts)


# 1. UI --------------------------------------

app_ui = (
    ui.page_navbar(
        # Page 1
        ui.nav_spacer(),  # push nav items to the right
        ui.nav_panel("Données",
            ui.page_sidebar(
                ui.sidebar(
                    ui.input_select(
                        "default_dataset",
                        "Choisir un jeu de données prédéfini:",
                        {
                            "iris": "Iris",
                            "penguins": "Palmer Penguins"
                        }
                    ),
                    ui.input_file("dataset_input", "Ou importer un fichier CSV ou Parquet", accept=[".csv", ".parquet"]),
                    position='right',
                    bg="#f8f8f8"
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Aperçu des données"),
                        ui.output_data_frame("data_view"),
                        full_screen=True,
                    ),
                    ui.card(
                        ui.card_header("Résumé statistique"),
                        ui.output_ui("data_summary"),
                        full_screen=True,
                    ),
                ),
            ),
        ),
        # Page 2
        ui.nav_panel("Préparer ses requêtes",
            ui.page_sidebar(
                ui.sidebar(
                    ui.input_file("request_input", "📂 Importer un fichier JSON", accept=[".json"]),
                    ui.input_action_button("save_json", "💾 Exporter le JSON"),
                    position="right",
                    bg="#f8f8f8"
                ),
                ui.panel_well(
                    ui.h4("➕ Ajouter une requête"),
                    ui.br(),
                    # Ligne 1
                    ui.row(
                        ui.column(3, ui.input_selectize("type_req", "Type de requête:", choices=type_req, selected=type_req[0])),
                        ui.column(3, ui.input_text("filtre", "Condition de filtrage:")),
                        ui.column(3, ui.input_text("borne_min", "Borne min:"))
                    ),
                    ui.br(),
                    # Ligne 2
                    ui.row(
                        ui.column(3, ui.input_selectize("variable", "Variable:", choices={}, options={"plugins": ["clear_button"]})),
                        ui.column(3, ui.input_selectize("group_by", "Regrouper par:", choices={}, multiple=True)),
                        ui.column(3, ui.input_text("borne_max", "Borne max:"))
                    ),
                    ui.br(),
                    # Ligne 3 (Optionnel)
                    ui.panel_conditional("input.type_req == 'Quantile' ",
                        ui.row(
                            ui.column(3, ui.input_numeric("alpha", "Ordre du quantile:", 0.5, min=0, max=1, step=0.01)),
                            ui.column(3, ui.input_text("candidat", "Liste des candidats:"))
                        ),
                    ),
                    ui.br(),
                    # Ligne 4 : bouton aligné à droite
                    ui.row(
                        ui.column(12,
                            ui.div(
                                ui.input_action_button("add_req", "➕ Ajouter la requête"),
                                class_="d-flex justify-content-end"
                            )
                        )
                    ),
                ),
                ui.hr(),
                ui.layout_columns(
                    ui.panel_well(
                        ui.h4("🗑️ Supprimer une requête"),
                        ui.br(),
                        # Ligne 1
                        ui.row(
                            ui.column(6, ui.input_selectize("delete_req", "Requêtes à supprimer:", choices={}, multiple=True)),
                        ),
                        # Ligne 2 : bouton aligné à droite
                        ui.row(
                            ui.column(12,
                                ui.div(
                                    ui.input_action_button("delete_btn", "Supprimer"),
                                    class_="d-flex justify-content-end"
                                )
                            )
                        ),
                    ),
                    ui.panel_well(
                        ui.h4("🗑️ Supprimer toutes les requêtes"),
                        ui.br(),
                        ui.row(
                            ui.column(12, ui.div(style="height: 85px"))  # Hauteur du champ texte simulée
                        ),
                        ui.row(
                            ui.column(12,
                                ui.div(
                                    ui.input_action_button("delete_all_btn", "Supprimer TOUT", class_="btn btn-danger"),
                                    class_="d-flex justify-content-end"
                                )
                            )
                        ),
                    ),
                ),
                ui.hr(),
                ui.panel_well(
                    ui.h4("📋 Requêtes actuelles"),
                    ui.br(),
                    ui.output_ui("req_display")
                ),
            )
        ),

        ui.nav_panel("Mécanisme DP",
            ui.panel_well(
                ui.h4("Mécanisme DP : ajout d'un bruit Gaussien centré (Comptage et Total)"),
                ui.br(),
                ui.div(
                    # Partie gauche : slider + résumé
                    ui.div(
                        ui.div(
                            ui.input_slider("scale_gauss", "Écart type du bruit :", min=1, max=100, value=10),
                            style="width: 400px;"
                        ),
                        ui.div(
                            ui.output_ui("interval_summary"),
                        ),
                        style="display: flex; flex-direction: column; gap: 20px;"
                    ),

                    # Partie gauche : deux tableaux côte à côte
                    ui.div(
                        ui.layout_column_wrap(
                            ui.card(
                                ui.card_header("Tableau de comptage non bruité"),
                                ui.output_data_frame("cross_table"),
                            ),
                            ui.card(
                                ui.card_header("Exemple après bruitage (sans post-traitement)"),
                                ui.output_data_frame("cross_table_2"),
                            ),
                            width=1 / 2,
                        ),
                        style="flex: 1;"
                    ),

                    # Partie budget DP : propre et sobre
                    ui.div(
                        ui.output_ui("dp_budget_summary"),
                        ui.input_slider(
                            "delta_slider",
                            "Exposant de δ",
                            min=-10,
                            max=-1,
                            value=-3,
                            step=1,
                            width="320px"
                        ),
                        style="margin-top: 30px; display: flex; flex-direction: column; gap: 16px;"
                    ),


                    style="display: flex; align-items: flex-start; gap: 50px;"
                ),
            ),
            ui.hr(),
            ui.panel_well(
                ui.h4("Mécanisme DP : scorer des candidats et tirer le score minimal après ajout d'un bruit (Quantile)"),
                ui.br(),
                # Conteneur principal en colonnes
                ui.div(
                    # Ligne 1
                    ui.div(
                        ui.layout_columns(
                            # Colonne 1 : texte explicatif
                            ui.div( # $\textrm{score}(x, c, \alpha_{num}, \alpha_{den}) = -|\alpha_{den} \cdot \#(x < c) - \alpha_{num} \cdot (|x| - \#(x = c))|$
                                ui.HTML("""
                                    <div style='margin-top:20px; padding:10px; background-color:#f9f9f9; border-radius:12px;
                                                font-family: "Raleway", "Garamond", sans-serif; font-size:16px; color:#333'>
                                        <p style="margin-bottom:10px">
                                            <strong>Exemple d'application à la variable <em>body_mass_g</em> du dataset Penguins :</strong>
                                        </p>
                                        <p style="margin-left:10px">
                                            La fonction de score utilisée est :
                                            <br><br>
                                            <strong>score</strong>(x, c, &alpha;) =<br>
                                            − 10 000 &times; | ∑<sub>i=1</sub><sup>n</sup> 1<sub>{x<sub>i</sub> &lt; c}</sub> − &alpha; &times; (n − ∑<sub>i=1</sub><sup>n</sup> 1<sub>{x<sub>i</sub> = c}</sub>) |
                                        </p>
                                        <p style="margin-left:10px">
                                            où <strong>α</strong> est l’ordre du quantile, et <strong>c</strong> un candidat et <strong>x</strong> notre variable d'intérêt de taille <strong>n</strong>.
                                        </p>
                                    </div>
                                    """),
                                style="padding: 10px;"
                            ),
                            # Colonne 2 : première carte
                            ui.card(
                                ui.card_header("Histogramme"),
                                ui.output_plot("histo_plot"),
                                full_screen=True,
                            ),
                            # Colonne 3 : deuxième carte
                            ui.card(
                                ui.card_header("Fonction de répartion empirique"),
                                ui.output_plot("fc_emp_plot"),
                                full_screen=True,
                            ),
                            col_widths=[3, 4, 5]
                        ),
                        style="margin-bottom: 40px;"
                    ),

                    # Ligne 2
                    ui.div(
                        ui.layout_columns(
                            # Colonne 1 : texte + sliders + équation
                            ui.div(
                                ui.HTML("<strong>Paramètres :</strong>"),
                                ui.input_slider("epsilon_slider", "Budget epsilon :", min=0.01, max=5, value=0.5, step=0.01),
                                ui.input_slider("alpha_slider", "Ordre du quantile :", min=0, max=1, value=0.5, step=0.01),
                                ui.p("Définir l’intervalle et le pas pour le candidat :"),
                                # Ligne horizontale pour les 3 champs
                                ui.div(
                                    ui.input_numeric("candidat_min", "Candidat min :", value=2500),
                                    ui.input_numeric("candidat_max", "Candidat max :", value=6500),
                                    ui.input_numeric("candidat_step", "Pas :", value=100),
                                    style="display: flex; flex-direction: row; gap: 15px; align-items: flex-end;"
                                ),
                                style="padding: 10px; display: flex; flex-direction: column; gap: 20px;"
                            ),
                            # Colonne 2 : carte placeholder
                            ui.card(
                                ui.card_header("Score des candidats"),
                                output_widget("score_plot"),
                                full_screen=True,
                            ),
                            # Colonne 3 : carte placeholder
                            ui.card(
                                ui.card_header("Probabilité de sélection"),
                                output_widget("proba_plot"),
                                full_screen=True,
                            ),
                            col_widths=[2, 6, 4]
                        )
                    ),
                    style="display: flex; flex-direction: column; gap: 40px;"
                )
            ),
        ),

        ui.nav_panel("Conception du budget", ui.h3("À compléter...")),

        ui.nav_panel("Résultat DP", ui.h3("À compléter...")),

        ui.nav_panel("A propos", ui.h3("À compléter...")),

        title=ui.div(
            ui.img(src="insee-logo.jpg", height="80px", style="margin-right:10px"),
            ui.div("Poc – Differential Privacy", style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 400; font-size: 28px;"),
            style="display: flex; align-items: center; gap: 10px;"
        ),
        id="page",
    )
)

# 2. Server ----------------------------------


def server(input, output, session):

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
    @render.ui
    def data_summary():
        return ui.HTML(dataset().describe(include="all").to_html())

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

    @reactive.Effect
    def update_variable_choices():
        # Met à jour dynamiquement les choix de la selectize input
        ui.update_selectize("variable", choices=variable_choices())
        ui.update_selectize("group_by", choices=variable_choices())
        ui.update_selectize("delete_req", choices=list(requetes().keys()))

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

        base_dict = {
            "type": input.type_req(),
            "variable": input.variable(),
            "bounds": [input.borne_min(), input.borne_max()],
            "by": input.group_by(),
            "filtre": input.filtre(),
        }

        if input.type_req() == 'Quantile':
            base_dict.update({
                "alpha": input.alpha(),
                "candidat": input.candidat(),
            })

        current[new_id] = {
            k: v for k, v in base_dict.items()
            if v not in [None, "", (), ["", ""]]
        }
        requetes.set(current)
        ui.notification_show(f"✅ Requête `{new_id}` ajoutée", type="message")

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

    @output
    @render.ui
    @reactive.event(input.request_input, input.add_req, input.delete_btn, input.delete_all_btn)
    def req_display():
        from src.process_tools import process_request
        import polars as pl

        data = requetes()
        if not data:
            return ui.p("Aucune requête chargée.")

        panels = []

        for key, req in data.items():
            # Colonne de gauche : paramètres
            df = pl.from_pandas(dataset()).lazy()
            resultat = process_request(df, req)

            if req.get("by") is not None:
                resultat = resultat.sort(by=req.get("by"))

            param_card = ui.card(
                ui.card_header("Paramètres"),
                make_card_body(req)
            )

            # Colonne de droite : table / placeholder
            result_card = ui.card(
                ui.card_header("Résultats sans application de la DP (à titre indicatif)"),
                ui.tags.style("""
                    .table {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        font-size: 0.9rem;
                        box-shadow: 0 0 10px rgba(0,0,0,0.05);
                        border-radius: 0.25rem;
                        border-collapse: collapse;
                        width: 100%;
                        border: 1px solid #dee2e6;
                    }
                    .table-hover tbody tr:hover {
                        background-color: #f1f1f1;
                    }
                    .table-striped tbody tr:nth-of-type(odd) {
                        background-color: #fafafa;
                    }
                    table.table thead th {
                        background-color: #f8f9fa !important;
                        font-weight: 700 !important;
                        border-left: 1px solid #dee2e6;
                        border-right: 1px solid #dee2e6;
                        border-bottom: 2px solid #dee2e6;
                        padding: 0.3rem 0.6rem;
                        vertical-align: middle !important;
                        text-align: center;
                        position: sticky;
                        top: 0;
                        z-index: 10;
                    }
                    tbody td {
                        padding: 0.3rem 0.6rem;
                        vertical-align: middle !important;
                        text-align: center;
                        border-left: 1px solid #dee2e6;
                        border-right: 1px solid #dee2e6;
                    }
                    thead th:first-child, tbody td:first-child {
                        border-left: none;
                    }
                    thead th:last-child, tbody td:last-child {
                        border-right: none;
                    }
                """),
                ui.HTML(resultat.to_pandas().to_html(
                    classes="table table-striped table-hover table-sm text-center align-middle",
                    border=0,
                    index=False
                )),
                height="300px",
                fillable = False,
                full_screen=True
            ),

            # Ligne avec deux colonnes côte à côte
            content_row = ui.row(
                ui.column(4, param_card),
                ui.column(8, result_card)
            )

            # Panneau d'accordéon contenant la ligne
            panels.append(
                ui.accordion_panel(f"{key} — {req.get('type', '—')}", content_row)
            )

        return ui.accordion(*panels)

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
        df = sns.load_dataset("penguins")
        table = df.groupby(["species", "island"]).size().unstack(fill_value=0)
        flat_table = table.reset_index().melt(id_vars="species", var_name="island", value_name="count")
        return flat_table.sort_values(by=["species", "island"])

    @output
    @render.data_frame
    @reactive.event(input.scale_gauss)
    def cross_table_2():
        from src.request_class import count_dp
        import opendp.prelude as dp
        import polars as pl
        dp.enable_features("contrib")

        key_values = {
            "species": ["Adelie", "Chinstrap", "Gentoo"],
            "island": ["Biscoe", "Dream", "Torgersen"]
        }
        df = sns.load_dataset("penguins")
        df_lazy = pl.from_pandas(df).lazy()

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
        from src.fonctions import eps_from_rho_delta
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
        df = sns.load_dataset("penguins")
        quantile_val = np.quantile(df['body_mass_g'].dropna(), input.alpha_slider())
        fig, ax = plt.subplots()
        sns.histplot(df['body_mass_g'], stat="percent", bins=40, color='lightcoral', ax=ax)
        # Ligne verticale pour le quantile
        ax.axvline(quantile_val, color='black', linestyle='--', linewidth=2)

        ax.set_xlabel("Valeur")
        ax.set_ylabel("Pourcentage")
        ax.grid(True)
        return fig

    @render.plot
    def fc_emp_plot():
        df = sns.load_dataset("penguins")
        sorted_df = np.sort(df['body_mass_g'])
        cdf = np.arange(1, len(df) + 1) / len(df)
        alpha = input.alpha_slider()
        quantile_val = np.quantile(df['body_mass_g'].dropna(), alpha)

        fig, ax = plt.subplots()

        ax.plot(sorted_df, cdf, color="lightcoral")
        # Segment vertical qui monte jusqu'à alpha
        ax.plot([quantile_val, quantile_val], [0, alpha], color='black', linestyle='--', linewidth=2)

        # Segment horizontal qui va jusqu'au quantile
        ax.plot([min(df['body_mass_g']), quantile_val], [alpha, alpha], color='black', linestyle='--', linewidth=2)

        ax.text(quantile_val, alpha +0.1, rf"$q_{{{input.alpha_slider():.2f}}} = {quantile_val:.0f}$",
                color='black', ha='right', va='center', fontsize=11)

        ax.set_xlabel("Valeur")
        ax.set_ylabel("Probabilité cumulative")
        ax.grid(True)
        return fig

    @render_widget
    def score_plot():
        from scipy.stats import gumbel_r
        import plotly.graph_objects as go

        df = sns.load_dataset("penguins")
        candidats = np.linspace(input.candidat_min(), input.candidat_max(), input.candidat_step() + 1).tolist()
        scores, sensi = manual_quantile_score(df['body_mass_g'], candidats, alpha=input.alpha_slider(), et_si=True)

        # Intervalle de confiance à 99%
        low_q, high_q = gumbel_r.ppf([0.005, 0.995], loc=0, scale=2 * sensi / input.epsilon_slider())
        lower = scores + low_q
        upper = scores + high_q

        min_idx = np.argmin(scores)
        min_lower = lower[min_idx]
        min_upper = upper[min_idx]

        # Séparer les points rouges et bleus
        rouges_x, rouges_y, rouges_err = [], [], []
        bleus_x, bleus_y, bleus_err = [], [], []

        for c, s, l, u in zip(candidats, scores, lower, upper):
            if not (u < min_lower or l > min_upper):
                rouges_x.append(c)
                rouges_y.append(s)
                rouges_err.append([s - l, u - s])
            else:
                bleus_x.append(c)
                bleus_y.append(s)
                bleus_err.append([s - l, u - s])

        # Trace bleue
        trace_bleu = go.Scatter(
            x=bleus_x,
            y=bleus_y,
            mode='markers',
            marker=dict(color='blue', size=10, opacity=0.7),
            name='Non chevauchement'
        )

        # Trace rouge
        trace_rouge = go.Scatter(
            x=rouges_x,
            y=rouges_y,
            mode='markers',
            marker=dict(color='red', size=10, opacity=0.7),
            name='Chevauchement'
        )

        # Barres d’erreur avec trace visible pour la légende
        trace_erreur = go.Scatter(
            x=candidats,
            y=scores,
            mode='markers',  # Affiche un petit point noir pour activer la légende
            marker=dict(color='black', size=2, symbol='line-ns-open', opacity=0.7),
            error_y=dict(
                type='data',
                symmetric=False,
                array=[u - s for s, u in zip(scores, upper)],
                arrayminus=[s - l for s, l in zip(scores, lower)],
                color='black',
                thickness=1,
                width=4
            ),
            name='IC à 99% (Gumbel)'
        )

        layout = go.Layout(
            xaxis=dict(
                title="Candidat",
                showgrid=True,
                gridcolor='lightgray',
                linecolor='black',
                linewidth=1,
                mirror=True
            ),
            yaxis=dict(
                title="Score",
                showgrid=True,
                gridcolor='lightgray',
                linecolor='black',
                linewidth=1,
                mirror=True
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                x=0.5, y=1.2,
                xanchor='center',
                yanchor='top',
                orientation='h',
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='lightgray',
                borderwidth=1
            ),
            margin=dict(t=50, r=30, l=60, b=60),
        )

        return go.Figure(data=[trace_erreur, trace_bleu, trace_rouge], layout=layout)

    @render_widget 
    def proba_plot():
        import plotly.graph_objects as go
        df = sns.load_dataset("penguins")
        candidats = np.linspace(input.candidat_min(), input.candidat_max(), input.candidat_step() + 1).tolist()
        scores, sensi = manual_quantile_score(df['body_mass_g'], candidats, alpha=input.alpha_slider(), et_si=True)

        # Probabilités
        proba_non_norm = np.exp(- input.epsilon_slider() * scores / (2 * sensi))
        proba = proba_non_norm / np.sum(proba_non_norm)

        # Seuil du 95e percentile
        seuil = np.quantile(proba, 0.95)

        # Points rouges
        trace_red = go.Scatter(
            x=[c for c, p in zip(candidats, proba) if p >= seuil],
            y=[p for p in proba if p >= seuil],
            mode='markers',
            marker=dict(color='red', size=10, opacity=0.7),
            name='≥ 95% des proba'
        )

        # Points bleus
        trace_blue = go.Scatter(
            x=[c for c, p in zip(candidats, proba) if p < seuil],
            y=[p for p in proba if p < seuil],
            mode='markers',
            marker=dict(color='blue', size=10, opacity=0.7),
            name='< 95% des proba'
        )

        layout = go.Layout(
            xaxis=dict(
                title="Candidat",
                showgrid=True,
                gridcolor='lightgray',
                linecolor='black',  # axe X en noir
                linewidth=1,
                mirror=True         # pour que le bas ET le haut aient une ligne noire
            ),
            yaxis=dict(
                title="Probabilité",
                showgrid=True,
                gridcolor='lightgray',
                linecolor='black',  # axe Y en noir
                linewidth=1,
                mirror=True         # pour que la gauche ET la droite aient une ligne noire
            ),
            plot_bgcolor='white',   # fond de la zone de traçage
            paper_bgcolor='white',  # fond global (autour du graphique)
            legend=dict(
                x=0.5, y=1.2,
                xanchor='center',
                yanchor='top',
                orientation='h',  # Horizontale
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='lightgray',
                borderwidth=1
            ),
            margin=dict(t=50, r=30, l=60, b=60),
        )

        return go.Figure(data=[trace_red, trace_blue], layout=layout)


www_dir = Path(__file__).parent / "www"

app = App(app_ui, server, static_assets=www_dir)
# shiny run --reload shiny_app.py
