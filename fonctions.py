import polars as pl
import numpy as np
from request_class import count_dp, mean_dp, sum_dp, quantile_dp


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
    if rho <= 0:
        raise ValueError("rho must be positive")

    return rho + 2 * np.sqrt(rho * np.log(1 / delta))


def process_request_dp(context_rho, context_eps, key_values, req):

    variable = req.get("variable")
    by = req.get("by")
    bounds = req.get("bounds")
    filtre = req.get("filtre")
    alpha = req.get("alpha")
    candidats = req.get("candidats")
    type_req = req["type"]

    mapping = {
            "count": lambda: count_dp(context_rho, key_values, by=by, variable=None, filtre=filtre),
            "mean": lambda: mean_dp(context_rho, key_values, by=by, variable=variable, bounds=bounds, filtre=filtre),
            "sum": lambda: sum_dp(context_rho, key_values, by=by, variable=variable, bounds=bounds, filtre=filtre),
            "quantile": lambda: quantile_dp(context_eps, key_values, by=by, variable=variable, bounds=bounds, filtre=filtre, alpha=alpha, candidats=candidats)
        }

    if type_req not in mapping:
        raise ValueError(f"Type de requête non supporté : {type_req}")

    return mapping[type_req]().execute()


def process_request(df: pl.LazyFrame, req: dict) -> pl.LazyFrame:

    variable = req.get("variable")
    by = req.get("by")
    bounds = req.get("bounds")
    alpha = req.get("alpha")
    type_req = req["type"]

    # Appliquer filtre si fourni (attention : sécurité si eval utilisé)
    # if filtre:
    #    try:
    #        df = df.filter(eval(f"pl.col('{filtre}')"))  # À sécuriser
    #    except Exception as e:
    #        print(f"Erreur dans le filtre '{filtre}': {e}")

    # Appliquer les bornes si variable et bounds sont présents
    if variable and bounds:
        L, U = bounds
        df = df.filter((pl.col(variable) >= L) & (pl.col(variable) <= U))

    # Traitement par type de requête
    if type_req == "count":
        if by:
            df = df.group_by(by).agg(pl.count().alias("count"))
        else:
            df = df.select(pl.count())

    elif type_req == "mean":
        if by:
            df = df.group_by(by).agg(pl.col(variable).mean().alias("mean"))
        else:
            df = df.select(pl.col(variable).mean().alias("mean"))

    elif type_req == "sum":
        if by:
            df = df.group_by(by).agg(pl.col(variable).sum().alias("sum"))
        else:
            df = df.select(pl.col(variable).sum().alias("sum"))

    elif type_req == "quantile":
        if alpha is None:
            alpha = [0.5]

        elif not isinstance(alpha, list):
            alpha = [alpha]

        if by:
            df = df.group_by(by).agg(
                pl.col(variable).quantile(alpha, interpolation="nearest").alias(f"quantile_{alpha}") for alpha in alpha
            )
        else:
            df = df.select(
                pl.col(variable).quantile(alpha, interpolation="nearest").alias(f"quantile_{alpha}") for alpha in alpha
            )

    else:
        raise ValueError(f"Type de requête inconnu : {type_req}")

    return df.collect()
