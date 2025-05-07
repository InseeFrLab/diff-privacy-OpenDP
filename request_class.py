from abc import ABC, abstractmethod
import opendp.prelude as dp
import polars as pl
import re
import operator

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
                .agg(dp.len())
                .join(self.generate_public_keys(by_keys=self.by), on=self.by, how="right")
            )
        else:
            query = query.select(dp.len())

        return query


class mean_dp(request_dp):
    def execute(self):
        l, u = self.bounds
        query = self.context.query()
        expr = (
            pl.col(self.variable)
            .fill_null(0)
            .dp.mean(bounds=(l, u))
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
    def __init__(self, context, key_values, by=None, variable=None, bounds=None, filtre=None, alpha=None):
        super().__init__(context, key_values, by, variable, bounds, filtre)

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
            .dp.quantile(a, range(0, 21))
            .alias(f"{a}-Quantile")
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
