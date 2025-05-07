import polars as pl
import opendp.prelude as dp


# Fonctions de traitement
def apply_filter(df, filtre):
    if filtre:
        return df.filter(eval(f'pl.col("{filtre}")'))
    return df


def process_request(df: pl.LazyFrame, req: dict) -> pl.LazyFrame:
    filtre = req.get("filtre")
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
            alpha = 0.5
        if by:
            df = df.group_by(by).agg(
                pl.col(variable).quantile(alpha, interpolation="nearest").alias(f"quantile_{alpha}")
            )
        else:
            df = df.select(
                pl.col(variable).quantile(alpha, interpolation="nearest").alias(f"quantile_{alpha}")
            )

    else:
        raise ValueError(f"Type de requête inconnu : {type_req}")

    return df.collect()


def process_request_dp(context, req):

    variable = req.get("variable")
    by = req.get("by")
    bounds = req.get("bounds")
    alpha = req.get("alpha")

    if variable and bounds:
        L, U = bounds

    type_req = req["type"]

    if type_req == "count":
        query = count_dp(context, by)

    elif type_req == "mean":
        query = mean_dp(context, by, variable, l=L, u=U)

    elif type_req == "sum":
        query = sum_dp(context, by, variable, l=L, u=U)

    elif type_req == "quantile":
        query = quantile_dp(context, by, variable, alpha)

    else:
        return f"Type de requête inconnu : {type_req}"

    # print(query.summarize())
    return query.release().collect()


def count_dp(context, by):
    query = (
        context.query()
        .group_by(by)
        .agg(dp.len())
    )

    return query


def mean_dp(context, by, variable, l, u):
    query = (
        context.query()
        .group_by(by)
        .agg(
            pl.col(variable)
            .fill_null(0)  # Obligatoire
            .dp.mean(bounds=(l, u))
        )
    )

    return query


def sum_dp(context, by, variable, l, u):
    query = (
        context.query()
        .group_by(by)
        .agg(
            pl.col(variable)
            .fill_null(0)
            .dp.sum((l, u))
        )
    )

    return query


def quantile_dp(context, by, variable, alpha):
    if alpha is None:
        alpha = 0.5  # default to median
        
    query = (
        context.query()
        .group_by(by)
        .agg(
            pl.col(variable)
            .fill_null(0)  # Obligatoire
            .dp.quantile(a, range(0, 21))
            .alias(f"{a}-Quantile")
            for a in [alpha]
        )
    )

    return query
