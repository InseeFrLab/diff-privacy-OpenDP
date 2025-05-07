import opendp.prelude as dp
from functools import lru_cache
dp.enable_features("contrib", "honest-but-curious")

d_in = 1


def find_dp_budget(transformation, budget, s):
    chain = transformation >> dp.m.then_gaussian(scale=s)
    return dp.c.make_fix_delta(dp.c.make_zCDP_to_approxDP(chain), delta=budget[1])


def return_dp(method, transformation, budget):
    if method == "gauss":
        scale = dp.binary_search_param(
            lambda s: find_dp_budget(transformation, budget, s),
            d_in=d_in, d_out=budget)
      
        print("Au seuil de 95%, étant donné le budget dépensé, le résultat est précis à plus ou moins près ",
            dp.gaussian_scale_to_accuracy(scale, alpha=0.05))

        return dp.binary_search_chain(
            lambda s: find_dp_budget(transformation, budget, s),
            d_in=d_in, d_out=budget)  # epsilon-delta
    else:
        scale = dp.binary_search_param(
            lambda s: transformation >> dp.m.then_laplace(scale=s),
            d_in=d_in, d_out=budget[0])

        print("Au seuil de 95%, étant donné le budget dépensé, le résultat est précis à plus ou moins près ",
            dp.laplacian_scale_to_accuracy(scale, alpha=0.05))

        return dp.binary_search_chain(
            lambda s: transformation >> dp.m.then_laplace(scale=s),
            d_in=d_in, d_out=budget[0],  # epsilon
            bounds=(0., 1e9))


@lru_cache(maxsize=None)
def make_count_with(*, budget, method="gauss"):

    if isinstance(budget, float):
        budget = (budget, 0.0)  # budget = (epsilon, delta)

    counter = (
        dp.t.make_split_dataframe(separator=",", col_names=col_names) >>
        dp.t.make_select_column(col_names[0], str) >>
        dp.t.then_count()
    )

    return return_dp(method, counter, budget)


@lru_cache(maxsize=None)
def make_sum_with(*, var, var_bounds, budget, method="gauss"):

    if isinstance(budget, float):
        budget = (budget, 0.0)  # budget = (epsilon, delta)

    bounded_sum = (
        dp.t.make_split_dataframe(separator=",", col_names=col_names) >>
        dp.t.make_select_column(key=var, TOA=str) >>
        dp.t.then_cast_default(TOA=float) >>
        dp.t.then_clamp(bounds=var_bounds) >>
        dp.t.then_sum()
    )

    return return_dp(method, bounded_sum, budget)


@lru_cache(maxsize=None)
def make_mean_with(*, target_size, var, var_bounds, prior, budget, method="gauss"):

    if isinstance(budget, float):
        budget = (budget, 0.0)  # budget = (epsilon, delta)

    mean_chain = (
        dp.t.make_split_dataframe(separator=",", col_names=col_names) >>
        dp.t.make_select_column(key=var, TOA=str) >>
        dp.t.then_cast_default(TOA=float) >>
        # Resize the dataset to length `target_size`.
        #     If there are fewer than `target_size` rows in the data, fill with a constant.
        #     If there are more than `target_size` rows in the data, only keep `data_size` rows
        dp.t.then_resize(size=target_size, constant=prior) >>
        dp.t.then_clamp(bounds=var_bounds) >>
        dp.t.then_mean()
    )

    return return_dp(method, mean_chain, budget)


@lru_cache(maxsize=None)
def make_variance_with(*, target_size, var, var_bounds, prior, budget, method="gauss"):

    if isinstance(budget, float):
        budget = (budget, 0.0)  # budget = (epsilon, delta)

    mean_chain = (
        dp.t.make_split_dataframe(separator=",", col_names=col_names) >>
        dp.t.make_select_column(key=var, TOA=str) >>
        dp.t.then_cast_default(TOA=float) >>
        # Resize the dataset to length `target_size`.
        #     If there are fewer than `target_size` rows in the data, fill with a constant.
        #     If there are more than `target_size` rows in the data, only keep `data_size` rows
        dp.t.then_resize(size=target_size, constant=prior) >>  # On peut choisir de donner la vrai size si ca n'est pas confidentiel
        dp.t.then_clamp(bounds=var_bounds) >>
        dp.t.then_variance()
    )

    return return_dp(method, mean_chain, budget)


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

if __name__ == "__main__":

    with open('data/data.csv') as input_file:
        data = input_file.read()

    col_names = ["age", "sex", "educ", "race", "income", "married"]
    budget = (1., 1e-7)
    income_bounds = (0., 150_000.)
    age_bounds = (0., 120.)  # an educated guess
    age_prior = 38.

    count_chain = make_count_with(budget=1., method="laplace")
    print(count_chain(data))
    count_chain = make_count_with(budget=budget, method="gauss")
    print(count_chain(data))

    sum_chain = make_sum_with(var="income", var_bounds=income_bounds, budget=1., method="laplace")
    print(sum_chain(data))
    sum_chain = make_sum_with(var="income", var_bounds=income_bounds, budget=budget, method="gauss")
    print(sum_chain(data))

    mean_chain = make_mean_with(var="age", target_size=count_chain(data), var_bounds=age_bounds, prior=age_prior, budget=1., method="laplace")
    print(mean_chain(data))
    mean_chain = make_mean_with(var="age", target_size=count_chain(data), var_bounds=age_bounds, prior=age_prior, budget=budget, method="gauss")
    print(mean_chain(data))
