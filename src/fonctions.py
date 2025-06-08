import re
import operator
import numpy as np
import polars as pl
from scipy.optimize import fsolve
import plotly.graph_objects as go
import plotly.colors as pc
import pandas as pd
from shiny import ui
import asyncio


def affichage_requete(requetes, dataset):
    from src.process_tools import process_request

    panels = []

    for key, req in requetes.items():
        # Colonne de gauche : paramètres
        df = pl.from_pandas(dataset).lazy()
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
            fillable=False,
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


async def affichage_requete_dp(context_rho, context_eps, key_values, requetes, progress, results_store):
    from src.process_tools import process_request_dp
    panels = []

    # Copie de l'état actuel du store
    current_results = results_store()

    for i, (key, req) in enumerate(requetes.items(), start=1):
        progress.set(i, message=f"Requête {key} — {req.get('type', '—')}", detail="Calcul en cours...")
        await asyncio.sleep(0.05)

        resultat_dp = process_request_dp(context_rho, context_eps, key_values, req).execute()
        df_result = resultat_dp.release().collect()

        if req.get("type") == "Moyenne":
            df_result = df_result.with_columns(mean=pl.col.total / pl.col.len)

        if req.get("by") is not None:
            df_result = df_result.sort(by=req.get("by"))
            first_col = df_result.columns[0]
            new_order = df_result.columns[1:] + [first_col]
            df_result = df_result.select(new_order)

        # ✅ Stocker le résultat sous forme pandas
        current_results[key] = df_result.to_pandas()

        param_card = ui.card(
            ui.card_header("Paramètres"),
            make_card_body(req)
        )

        result_card = ui.card(
            ui.card_header("Résultats après application de la DP"),
            ui.HTML(df_result.to_pandas().to_html(
                classes="table table-striped table-hover table-sm text-center align-middle",
                border=0,
                index=False
            )),
            height="300px",
            fillable=False,
            full_screen=True
        )

        content_row = ui.row(
            ui.column(4, param_card),
            ui.column(8, result_card)
        )

        panels.append(
            ui.accordion_panel(f"{key} — {req.get('type', '—')}", content_row)
        )

    # ✅ Mise à jour finale du reactive.Value
    results_store.set(current_results)

    return ui.accordion(*panels)



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


def rho_from_eps_delta(epsilon, delta):
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    log_term = np.log(1 / delta)
    sqrt_term = np.sqrt(log_term * (epsilon + log_term))
    rho = 2 * log_term + epsilon - 2 * sqrt_term
    return rho


def eps_from_rho_delta(rho, delta):
    if rho < 0:
        raise ValueError("rho must be positive")

    def equation(y, rho, delta):
        return y - (rho + 2 * np.sqrt(rho * np.log(1 / (delta * (1 + (y - rho) / (2 * rho))))))

    epsilon_base = rho + 2 * np.sqrt(rho * np.log(1 / delta))
    y0 = rho + 1
    epsilon_small = fsolve(equation, y0, args=(rho, delta))[0]
    # print("Diff formule de base :", epsilon_base - epsilon_small)

    return epsilon_small


# Map des opérateurs Python vers leurs fonctions correspondantes
OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}


def parse_single_condition(condition: str) -> pl.Expr:
    """Transforme une condition string comme 'age > 18' en pl.Expr."""
    for op_str, op_func in OPS.items():
        if op_str in condition:
            left, right = condition.split(op_str, 1)
            left = left.strip()
            right = right.strip()
            # Gère les chaînes entre guillemets simples ou doubles
            if re.match(r"^['\"].*['\"]$", right):
                right = right[1:-1]
            elif re.match(r"^\d+(\.\d+)?$", right):  # nombre
                right = float(right) if '.' in right else int(right)
            return op_func(pl.col(left), right)
    raise ValueError(f"Condition invalide : {condition}")


def parse_filter_string(filter_str: str) -> pl.Expr:
    """Transforme une chaîne de filtres combinés en une unique pl.Expr."""
    # Séparation sécurisée via regex avec maintien des opérateurs binaires
    tokens = re.split(r'(\s+\&\s+|\s+\|\s+)', filter_str)
    exprs = []
    ops = []

    for token in tokens:
        token = token.strip()
        if token == "&":
            ops.append("&")
        elif token == "|":
            ops.append("|")
        elif token:  # une condition
            exprs.append(parse_single_condition(token))

    # Combine les expressions avec les bons opérateurs
    expr = exprs[0]
    for op, next_expr in zip(ops, exprs[1:]):
        if op == "&":
            expr = expr & next_expr
        elif op == "|":
            expr = expr | next_expr

    return expr


def format_scientifique(val, precision=2):
    base, exposant = f"{val:.{precision}e}".split("e")
    exposant = int(exposant)  # conversion propre de l'exposant
    return f"10^{exposant}"


def construire_dataframe_comparatif(resultats_reels, resultats_dp, requetes):
    lignes = []

    for key in resultats_reels.keys():
        req = requetes.get(key, {})
        type_req = req.get("type")
        alpha = req.get("alpha", [])
        if isinstance(alpha, float):
            alpha = [alpha]  # uniformise

        df_reel = resultats_reels[key]
        dp_info = resultats_dp.get(key)
        if dp_info is None or isinstance(df_reel, Exception) or isinstance(dp_info["data"], Exception):
            continue

        df_dp = dp_info["data"]
        summary = dp_info.get("summary", None)
        if summary is not None:
            scales = summary.select("scale").to_series().to_list()
        else:
            scales = [None]

        n_lignes = max(len(df_reel), len(df_dp))

        if type_req == "quantile":
            for j in range(n_lignes):
                sous_requete = chr(97 + j) if n_lignes > 1 else "a"
                for alpha_val in alpha:
                    col_name = f"quantile_{round(alpha_val, 3)}"

                    val_reelle = (
                        df_reel[j][col_name].item()
                        if j < len(df_reel) and col_name in df_reel.columns else None
                    )
                    val_dp = (
                        df_dp[j][col_name].item()
                        if j < len(df_dp) and col_name in df_dp.columns else None
                    )

                    lignes.append({
                        "requête": key,
                        "sous_requête": sous_requete,
                        "type_requête": col_name,
                        "valeur_réelle": val_reelle,
                        "valeur_DP": val_dp,
                        "écart_absolu": abs(val_reelle - val_dp) if val_reelle is not None and val_dp is not None else None,
                        "écart_relatif": 100 * abs(val_reelle - val_dp) / abs(val_reelle) if val_reelle not in (0, None) and val_dp is not None else None,
                        "scale": scales if len(scales) > 1 else [scales[0]],
                    })

        else:
            for j in range(n_lignes):
                sous_requete = chr(97 + j) if n_lignes > 1 else "a"

                val_reelle = (
                    df_reel[j][type_req].item()
                    if j < len(df_reel) and type_req in df_reel.columns else None
                )
                val_dp = (
                    df_dp[j][type_req].item()
                    if j < len(df_dp) and type_req in df_dp.columns else None
                )

                lignes.append({
                    "requête": key,
                    "sous_requête": sous_requete,
                    "type_requête": type_req,
                    "valeur_réelle": val_reelle,
                    "valeur_DP": val_dp,
                    "écart_absolu": abs(val_reelle - val_dp) if val_reelle is not None and val_dp is not None else None,
                    "écart_relatif": 100*abs(val_reelle - val_dp) / abs(val_reelle) if val_reelle not in (0, None) and val_dp is not None else None,
                    "scale": scales if len(scales) > 1 else [scales[0]],
                })

    return pl.DataFrame(lignes)


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

    if et_si:
        alpha_num = int(np.floor(alpha * 10_000))
        alpha_denum = 10_000

    scores = []
    for c in candidats:
        n_less = np.sum(data < c)
        n_equal = np.sum(data == c)
        score = alpha_denum * n_less - alpha_num * (len(data) - n_equal)
        scores.append(abs(score))

    return np.array(scores), max(alpha_num, alpha_denum - alpha_num)


# Fonction de création de barplot
def add_bar(fig, df, row, col, x_col, y_col, color):
    if not df.empty:
        bar_args = dict(
            x=df[x_col],
            y=df[y_col],
            marker=dict(
                color=color,
                line=dict(width=0),
                opacity=0.85, 
                cornerradius="15%"
            ),
            text=[
                f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
                for v in df[y_col]
            ],
            hovertemplate=(
                f"<b>%{{text}}</b>"
                + (f"<br>{x_col}: %{{x:.2f}}" if pd.api.types.is_numeric_dtype(df[x_col]) else f"<br>{x_col}: %{{x}}")
                + (f"<br>{y_col}: %{{y:.2f}}" if pd.api.types.is_numeric_dtype(df[y_col]) else f"<br>{y_col}: %{{y}}")
                + "<extra></extra>"
            ),
            textposition="auto",
            textfont=dict(color="white")
        )

        fig.add_trace(
            go.Bar(**bar_args),
            row=row,
            col=col
        )

    else:
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="text",
                text=["Aucune donnée"],
                textposition="middle center",
                showlegend=False
            ),
            row=row,
            col=col
        )


def add_scatter(fig, df, row, col, x_col, y_col, size_col=None):
    if not df.empty:
        # Génère une couleur unique pour chaque 'clé'
        unique_keys = df["requête"].unique()
        color_palette = pc.qualitative.Plotly
        color_map = {k: color_palette[i % len(color_palette)] for i, k in enumerate(unique_keys)}

        # Convertir les tailles en float si demandé
        sizes = (
            df[size_col].astype(float).to_numpy()
            if size_col else
            None
        )

        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="markers",
                marker=dict(
                    size=sizes if size_col else 10,
                    sizemode="area",
                    sizeref=2. * max(sizes) / (40.**2) if size_col else 1,
                    sizemin=5,
                    color=[color_map[k] for k in df["requête"]],
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                text=df["requête"],
                hovertemplate=(
                    f"<b>%{{text}}</b>"
                    + (f"<br>{x_col}: %{{x:.2f}}" if pd.api.types.is_numeric_dtype(df[x_col]) else f"<br>{x_col}: %{{x}}")
                    + (f"<br>{y_col}: %{{y:.2f}}" if pd.api.types.is_numeric_dtype(df[y_col]) else f"<br>{y_col}: %{{y}}")
                    + (f"<br>{size_col}: %{{marker.size:.2f}}" if size_col and size_col != x_col and size_col != y_col else "")
                    + "<extra></extra>"
                )
            ),
            row=row,
            col=col
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="text",
                text=["Aucune donnée"],
                textposition="middle center",
                showlegend=False
            ),
            row=row,
            col=col
        )
