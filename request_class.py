from abc import ABC, abstractmethod
import opendp.prelude as dp
import polars as pl
from fonctions import parse_filter_string


# Classe mère
class request_dp(ABC):
    def __init__(self, context, key_values, by=None, variable=None, bounds=None, filtre=None):
        self.context = context
        self.key_values = key_values
        self.by = by
        self.variable = variable
        self.bounds = bounds
        self.filtre = filtre

    def generate_public_keys(self, by_keys: list[str]) -> pl.LazyFrame:
        from itertools import product

        # Ne garder que les colonnes utiles pour le group_by
        values = [self.key_values[key] for key in by_keys if key in self.key_values]

        # Produit cartésien des valeurs
        combinaisons = list(product(*values))

        # Construction du LazyFrame public
        public_keys = pl.DataFrame([dict(zip(by_keys, comb)) for comb in combinaisons]).lazy()
        return public_keys

    @abstractmethod
    def execute(self):
        pass


class count_dp(request_dp):
    def execute(self):
        query = self.context.query()

        if self.filtre is not None:
            query = query.filter(parse_filter_string(self.filtre))

        if self.by is not None:
            query = (
                query.group_by(self.by)
                .agg(dp.len().alias("count"))
                .join(self.generate_public_keys(by_keys=self.by), on=self.by, how="right")
            )
        else:
            query = query.select(dp.len().alias("count"))

        return query


class mean_dp(request_dp):
    def execute(self):
        l, u = self.bounds
        query = self.context.query()
        expr = (
            pl.col(self.variable)
            .fill_null(0)
            .dp.mean(bounds=(l, u))
            .alias("mean")
        )

        if self.filtre is not None:
            query = query.filter(parse_filter_string(self.filtre))

        if self.by is not None:
            query = (
                query.group_by(self.by)
                .agg(expr)
                .join(self.generate_public_keys(by_keys=self.by), on=self.by, how="right")
            )
        else:
            query = query.select(expr)

        return query


class sum_dp(request_dp):
    def execute(self):
        l, u = self.bounds
        query = self.context.query()
        expr = (
            pl.col(self.variable)
            .fill_null(0)
            .dp.sum((l, u))
            .alias("sum")
        )

        if self.filtre is not None:
            query = query.filter(parse_filter_string(self.filtre))

        if self.by is not None:
            query = (
                query.group_by(self.by)
                .agg(expr)
                .join(self.generate_public_keys(by_keys=self.by), on=self.by, how="right")
            )
        else:
            query = query.select(expr)

        return query


class quantile_dp(request_dp):
    def __init__(self, context, key_values, variable, candidats, by=None, bounds=None, filtre=None, alpha=None):
        super().__init__(context, key_values, by, variable, bounds, filtre)

        # Attention, il y a y un problème avec OpenDP au niveau des candidats
        # Il faut qu'ils soient ordonnées de manière croissante ET de partie entière distincte
        if candidats["type"] == "list":
            self.candidats = candidats["values"]
        elif candidats["type"] == "range":
            import numpy as np
            self.candidats = np.arange(candidats["min"], candidats["max"] + candidats["step"], candidats["step"]).tolist()

        if alpha is None:
            self.list_alpha = [0.5]
        elif isinstance(alpha, list):
            self.list_alpha = alpha
        else:
            self.list_alpha = [alpha]

    def execute(self):
        query = self.context.query()
        aggs = [
            pl.col(self.variable)
            .fill_null(0)
            .dp.quantile(a, self.candidats)
            .alias(f"quantile_{a}")
            for a in self.list_alpha
        ]

        if self.filtre is not None:
            query = query.filter(parse_filter_string(self.filtre))

        if self.by is not None:
            query = (
                query.group_by(self.by)
                .agg(*aggs)
                .join(self.generate_public_keys(by_keys=self.by), on=self.by, how="right")
            )
        else:
            query = query.select(*aggs)

        return query
