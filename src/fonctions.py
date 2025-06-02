import re
import operator
import numpy as np
import polars as pl
from scipy.optimize import fsolve


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
    print("Diff formule de base :", epsilon_base - epsilon_small)

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
                        "écart_relatif": abs(val_reelle - val_dp) / abs(val_reelle) if val_reelle not in (0, None) and val_dp is not None else None,
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
                    "écart_relatif": abs(val_reelle - val_dp) / abs(val_reelle) if val_reelle not in (0, None) and val_dp is not None else None,
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
