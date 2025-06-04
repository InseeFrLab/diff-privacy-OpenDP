# Imports
import json
from shiny import App, ui, render, reactive
import pandas as pd
from pathlib import Path

# --- Etat de session réactif ---
requetes = reactive.Value({})  # Dictionnaire des requêtes
json_source = reactive.Value("")

type_req = ["Comptage", "Total", "Moyenne", "Quantile"]


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
                            ui.column(6, ui.input_text("delete_req_id", "ID de la requête (ex: req_1)")),
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

        ui.nav_panel("Mécanisme DP", ui.h3("À compléter...")),

        ui.nav_panel("Conception du budget", ui.h3("À compléter...")),

        ui.nav_panel("Modélisation", ui.h3("À compléter...")),

        ui.nav_panel("Export / Rapport", ui.h3("À compléter...")),

        title="POC - Differential Privacy",
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
            from seaborn import load_dataset
            return load_dataset(input.default_dataset())

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
        target = input.delete_req_id().strip()
        if target in current:
            del current[target]
            requetes.set(current)
            ui.notification_show(f"🗑️ Requête `{target}` supprimée", type="warning")
        else:
            ui.notification_show(f"❌ `{target}` introuvable", type="error")

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
        data = requetes()
        if not data:
            return ui.p("Aucune requête chargée.")

        rows = []
        row = []

        for i, (key, req) in enumerate(data.items()):
            panel = ui.card(
                ui.card_header(f"{key} — {req.get('type', '—')}"),
                make_card_body(req)
            )
            row.append(ui.column(4, panel))  # 3 colonnes de 4 → 12 colonnes max par ligne
            if len(row) == 3:
                rows.append(ui.row(*row))
                row = []

        if row:
            rows.append(ui.row(*row))

        return ui.div(*rows)


app = App(app_ui, server)
