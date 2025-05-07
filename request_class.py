from abc import ABC, abstractmethod
import opendp.prelude as dp
import polars as pl


# Classe mère
class request_dp(ABC):
    def __init__(self, context, by=None, variable=None, bounds=None, alpha=None):
        self.context = context
        self.by = by
        self.variable = variable
        self.bounds = bounds
        self.alpha = alpha

    @abstractmethod
    def execute(self):
        pass


class count_dp(request_dp):
    def execute(self):
        return (
            self.context.query()
            .group_by(self.by)
            .agg(
                dp.len()
            )
        )


class mean_dp(request_dp):
    def execute(self):
        l, u = self.bounds
        return (
            self.context.query()
            .group_by(self.by)
            .agg(
                pl.col(self.variable)
                .fill_null(0)
                .dp.mean(bounds=(l, u))
            )
        )


class sum_dp(request_dp):
    def execute(self):
        l, u = self.bounds
        return (
            self.context.query()
            .group_by(self.by)
            .agg(
                pl.col(self.variable)
                .fill_null(0)
                .dp.sum((l, u))
            )
        )


class quantile_dp(request_dp):
    def execute(self):
        alpha = self.alpha if self.alpha is not None else 0.5
        return (
            self.context.query()
            .group_by(self.by)
            .agg(
                pl.col(self.variable)
                .fill_null(0)
                .dp.quantile(alpha, range(0, 21))
                .alias(f"{alpha}-Quantile")
            )
        )
