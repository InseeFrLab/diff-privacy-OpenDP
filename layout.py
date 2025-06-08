from shiny import ui
from shinywidgets import output_widget

regions_france = [
    "France entière",
    "Auvergne-Rhône-Alpes",
    "Bourgogne-Franche-Comté",
    "Bretagne",
    "Centre-Val de Loire",
    "Corse",
    "Grand Est",
    "Hauts-de-France",
    "Île-de-France",
    "Normandie",
    "Nouvelle-Aquitaine",
    "Occitanie",
    "Pays de la Loire",
    "Provence-Alpes-Côte d’Azur",
    "Guadeloupe",
    "Martinique",
    "Guyane",
    "La Réunion",
    "Mayotte"
]

dataset = [
    "Fidéli",
    "Filosofi",
    "Penguin"
]


# Page Données ----------------------------------

def page_donnees():
    return ui.nav_panel("Données",
        ui.page_sidebar(
            sidebar_donnees(),
            layout_donnees()
        )
    )


def sidebar_donnees():
    return ui.sidebar(
        ui.input_select(
            "default_dataset",
            "Choisir un jeu de données prédéfini:",
            {"penguins": "Palmer Penguins"}
        ),
        ui.input_file("dataset_input", "Ou importer un fichier CSV ou Parquet", accept=[".csv", ".parquet"]),
        position='right',
        bg="#f8f8f8"
    )


def layout_donnees():
    return ui.navset_card_underline(
        ui.nav_panel("Aperçu des données",
            ui.output_data_frame("data_view"),
        ),
        ui.nav_panel("Résumé statistique",
            ui.output_data_frame("data_summary"),
        )
    )


# Page Mécanisme DP ----------------------------------

def page_mecanisme_dp():
    return ui.nav_panel("Mécanisme DP",
        bloc_bruit_gaussien(),
        ui.hr(),
        bloc_score_quantile()
    )


def bloc_bruit_gaussien():
    return ui.panel_well(
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
    )


def bloc_score_quantile():
    return ui.panel_well(
        ui.h4("Mécanisme DP : scorer des candidats et tirer le score minimal après ajout d'un bruit (Quantile)"),
        ui.br(),
        # Conteneur principal en colonnes
        ui.div(
            # Ligne 1
            ui.div(
                ui.layout_columns(
                    # Colonne 1 : texte explicatif
                    ui.div(
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
                            ui.input_slider("candidat_slider", "Intervalles des candidats", min=0, max=10000, value=[2500, 6500]),
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
    )


# Page Préparer ses requêtes ----------------------------------

def page_preparer_requetes():
    return ui.nav_panel("Préparer ses requêtes",
        ui.page_sidebar(
            sidebar_requetes(),
            bloc_ajout_requete(),
            ui.hr(),
            layout_suppression_requetes(),
            ui.hr(),
            bloc_requetes_actuelles()
        )
    )


def sidebar_requetes():
    return ui.sidebar(
        ui.input_file("request_input", "📂 Importer un fichier JSON", accept=[".json"]),
        ui.download_button("download_json", "💾 Télécharger les requêtes (JSON)", class_="btn-outline-primary"),
        position="right",
        bg="#f8f8f8"
    )


def bloc_ajout_requete():
    return ui.panel_well(
        ui.h4("➕ Ajouter une requête"),
        ui.br(),
        ui.row(
            ui.column(3, ui.input_selectize("type_req", "Type de requête:", choices=["Comptage", "Total", "Moyenne", "Quantile"], selected="Comptage")),
            ui.column(3, ui.input_text("filtre", "Condition de filtrage:")),
            ui.column(3, ui.input_text("borne_min", "Borne min:"))
        ),
        ui.br(),
        ui.row(
            ui.column(3, ui.input_selectize("variable", "Variable:", choices={}, options={"plugins": ["clear_button"]})),
            ui.column(3, ui.input_selectize("group_by", "Regrouper par:", choices={}, multiple=True)),
            ui.column(3, ui.input_text("borne_max", "Borne max:"))
        ),
        ui.br(),
        ui.panel_conditional("input.type_req == 'Quantile'",
            ui.row(
                ui.column(3, ui.input_numeric("alpha", "Ordre du quantile:", 0.5, min=0, max=1, step=0.01)),
                ui.column(3, ui.input_text("candidat", "Liste des candidats:"))
            ),
        ),
        ui.br(),
        ui.row(
            ui.column(12,
                ui.div(
                    ui.input_action_button("add_req", "➕ Ajouter la requête"),
                    class_="d-flex justify-content-end"
                )
            )
        )
    )


def layout_suppression_requetes():
    return ui.layout_columns(
        ui.panel_well(
            ui.h4("🗑️ Supprimer une requête"),
            ui.br(),
            ui.row(
                ui.column(6, ui.input_selectize("delete_req", "Requêtes à supprimer:", choices={}, multiple=True))
            ),
            ui.row(
                ui.column(12,
                    ui.div(
                        ui.input_action_button("delete_btn", "Supprimer"),
                        class_="d-flex justify-content-end"
                    )
                )
            )
        ),
        ui.panel_well(
            ui.h4("🗑️ Supprimer toutes les requêtes"),
            ui.br(),
            ui.row(ui.column(12, ui.div(style="height: 85px"))),
            ui.row(
                ui.column(12,
                    ui.div(
                        ui.input_action_button("delete_all_btn", "Supprimer TOUT", class_="btn btn-danger"),
                        class_="d-flex justify-content-end"
                    )
                )
            )
        )
    )


def bloc_requetes_actuelles():
    return ui.panel_well(
        ui.h4("📋 Requêtes actuelles"),
        ui.br(),
        ui.output_ui("req_display")
    )


# Page Conception du budget DP ----------------------------------

def page_conception_budget():
    return ui.nav_panel("Conception du budget",
        ui.page_sidebar(
            sidebar_budget(),
            bloc_budget_comptage(),
            ui.hr(),
            bloc_budget_total_moyenne(),
            ui.hr(),
            bloc_budget_quantile()
        )
    )


def sidebar_budget():
    return ui.sidebar(
        ui.h3("Définition du budget"),
        ui.input_numeric("budget_total", "Budget total :", 0.2, min=0.1, max=1, step=0.01),
        ui.input_selectize("echelle_geo", "Echelle géographique de l'étude:", choices=regions_france, selected="France entière"),
        ui.input_selectize("dataset_name", "Nom du dataset:", choices=dataset, selected="Penguin", options={"create": True}),
        ui.input_action_button("valider_budget", "Valider le budget DP"),
        position="right",
        bg="#f8f8f8"
    )


def bloc_budget_comptage():
    return ui.panel_well(
        ui.layout_columns(
            ui.card(
                ui.card_header("Répartition du budget pour les comptages"),
                ui.output_ui("radio_buttons_comptage"),
            ),
            ui.card(
                output_widget("plot_comptage"),
                full_screen=True
            ),
            col_widths=[4, 8]
        )
    )


def bloc_budget_total_moyenne():
    return ui.panel_well(
        ui.layout_columns(
            ui.card(
                ui.card_header("Répartition du budget pour les totaux et les moyennes"),
                ui.output_ui("radio_buttons_total_moyenne"),
            ),
            ui.card(
                output_widget("plot_total_moyenne"),
                full_screen=True
            ),
            col_widths=[4, 8]
        )
    )


def bloc_budget_quantile():
    return ui.panel_well(
        ui.layout_columns(
            ui.card(
                ui.card_header("Répartition du budget pour les quantiles"),
                ui.output_ui("radio_buttons_quantile"),
            ),
            ui.card(
                output_widget("plot_quantile"),
                full_screen=True
            ),
            col_widths=[4, 8]
        )
    )


# Page Résultat dp ----------------------------------

def page_resultat_dp():
    return ui.nav_panel("Résultat DP",
        ui.panel_well(
            ui.h4("Résultat des requêtes DP"),
            ui.br(),
            ui.div("Voici les résultats des requêtes différentes privées."),
            ui.download_button("download_xlsx", "💾 Télécharger les résultats (XLSX)", class_="btn-outline-primary"),
            ui.br(),
            ui.output_ui("req_dp_display")
        )
    )


# Page Etat budget dataset ----------------------------------

def page_etat_budget_dataset():
    return ui.nav_panel("Etat budget dataset",
        ui.card(
            ui.card_header("Aperçu des budgets dépensées"),
            ui.output_data_frame("data_budget_view")
        ),
        ui.output_ui("budget_display")
    )


# Page A propos ----------------------------------

def page_a_propos():
    return ui.nav_panel("A propos",
        ui.h3("À compléter...")
    )
