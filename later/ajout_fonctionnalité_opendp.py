import polars as pl
import opendp.prelude as dp
from opendp.extras.polars import DPExpr

dp.enable_features("contrib")


def my_custom_mean(self, bounds: tuple[float, float], taille: float, scale: float | None = None):
    # Exemple : retourner le type de l'expression
    return self.sum(bounds, scale) / taille


DPExpr.my_custom_mean = my_custom_mean

print([m for m in dir(DPExpr) if not m.startswith("_")])


context = dp.Context.compositor(
    # Many columns contain mixtures of strings and numbers and cannot be parsed as floats,
    # so we'll set `ignore_errors` to true to avoid conversion errors.
    data=pl.scan_csv(dp.examples.get_france_lfs_path(), ignore_errors=True),
    privacy_unit=dp.unit_of(contributions=36),
    privacy_loss=dp.loss_of(epsilon=1.0),
    split_evenly_over=10,
    margins=[
        dp.polars.Margin(
            # the biggest (and only) partition is no larger than
            #    France population * number of quarters
            max_partition_length=60_000_000 * 36
        )
    ],
)

query_work_hours = (
    # 99 represents "Not applicable"
    context.query()
    .select(
        pl.col.HWUSUAL
        .cast(int)
        .fill_null(35)
        .dp.my_custom_mean(bounds=(0, 80), taille=200000))
)

print(query_work_hours.release().collect())
