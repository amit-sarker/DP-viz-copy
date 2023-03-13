from collections import defaultdict
from typing import NamedTuple, Tuple, Dict, List

import numpy as np
import pandas as pd
# Local imports
from mbi import Domain, Dataset, FactoredInference
from scipy import sparse

import error
import mechanisms
import workload_builder as builder
from ektelo import matrix


class CacheEntry(NamedTuple):
    eps: float
    strategy: matrix.Kronecker
    At_y: np.ndarray


class BackEnd:
    def __init__(self, dataset: Dataset, budget: float, seed=0):
        self.dataset = dataset
        self.cache = defaultdict(list)
        self.budget = budget  # not in use now
        self.prng = np.random.RandomState(seed)                                # there was a seed before
        # self.prng = np.random.seed(1234)
        self.budget_spent = 0

    def measure_hdmm(self, workload: builder.Workload, eps: float, restarts: int, prng=None, seed=0):
        """Runs HDMM on the workload with the specified 'eps', adds the results
        to self.cache

        Args:
           workload: Workload matrix to be measured.
           eps: Privacy parameter.
           seed (int, optional): Seed for prng.
           restarts (int): Number of iterations to run optimization for

        Returns:
           Nothing, stores result in dict ``self.cache``.
        """
        # assert (
        #     self.budget > self.budget_spent + eps
        # ), "All available privacy budget has been spent"
        self.budget_spent += eps
        if prng:
            engine = mechanisms.HDMM(
                workload.matrix, self.dataset.datavector(), eps, prng=prng
            )

        if not prng:
            engine = mechanisms.HDMM(
                workload.matrix, self.dataset.datavector(), eps, prng=self.prng
            )

        # if prng is not None:
        #     print('custom seed', prng.laplace())
        #     engine = mechanisms.HDMM(
        #         workload.matrix, self.dataset.datavector(), eps, prng=prng
        #     )
        # else:
        #     print('default seed', self.prng.laplace())
        #     engine = mechanisms.HDMM(
        #         workload.matrix, self.dataset.datavector(), eps, prng=self.prng
        #     )

        engine.optimize(restarts=restarts)
        x_est, y_hat, strategy_matrix = engine.run(seed)
        # print(strategy_matrix.matrix.shape)
        # err = error.per_query_error(workload.matrix, engine.strategy, eps)
        W_ans = workload.matrix @ x_est

        key = tuple(sorted((workload.column_spec.items())))
        self.cache[key].append(CacheEntry(eps, engine.strategy, x_est))

        return y_hat, strategy_matrix

    def measure_pgm(self, marginals, iters, eps=1.0):
        """Runs PGM using measurements from Identity workloads specified in
           ``marginals``.

        Args:
           marginals (iterable): Iterable where each element is an iterable of
               containing the columns to perform Identity workloads over.
           iters (int): Number of iterations to run Mirror Descent (MD)
               algorithm for.
           eps (float): Privacy parameter.

        Returns:
           mbi.FactoredInference object holding learned graphical model.

        """
        assert (
            self.budget >= self.budget_spent + eps
        ), "All available privacy budget has been spent"
        self.budget_spent += eps

        scale = len(marginals) / eps
        noisy_answers = []

        # Measure noisy marginals using laplace
        for query in marginals:
            answer = self.dataset.project(query).datavector()
            y_hat = answer + np.random.laplace(loc=0, scale=scale, size=answer.size)
            assert y_hat.size == answer.size
            noisy_answers.append(y_hat)

        measurements = []
        # Fit a PGM to these noisy measurements
        for y_hat, query in zip(noisy_answers, marginals):
            marginal_matrix = sparse.identity(y_hat.size)
            measurements.append((marginal_matrix, y_hat, scale, query))

        engine = FactoredInference(self.dataset.domain, log=True, iters=iters)
        model = engine.estimate(measurements, engine="MD")
        return model

    def display(self, workload: builder.Workload, column_names=None):
        """Returns a visualization specification for ``workload``, the already
           measured workload. If ``workload`` has not mean measured, a KeyError
           will be thrown.

        Args:
           workload: Workload matrix to be displayed.

        Returns:
           DataFrame holding the visualization specification.
        """
        key = tuple(sorted(workload.column_spec.items()))
        cached_measurements = cache_search(self.cache, key)  # first check this measurements are returning legitimate outputs
        if not cached_measurements:
            raise KeyError(
                "Attempting to display results of a workload that has \
                    not yet been measured. Workload key: {}".format(
                    key
                )
            )

        true_answer = workload.matrix @ self.dataset.datavector()
        combined_answer, combined_error = remeasure(workload, cached_measurements) # figure out the outputs that does make sense
        return specification(
            combined_answer,
            true_answer,
            combined_error,
            workload.column_spec,
            self.dataset.domain.config,
            column_names=column_names,
        ), cached_measurements  # JOIE: might want specification to be a NamedTuple also


def cache_search(cache, target_key):
    """
    Searches cached measurements for matches to target_key.

    Returns:
        List of cached measurements which match 'target_key'. If there are no matches,
        the list is empty.

    Args:
        cache (dict): A dictionary where the key is the name of the columns measured
            and the value is the cached measurement over the columns.
        target_key ([tuple]): A list of tuples of the form (column_name, bin_width). These
            are the columns that the user currently wants to measure over
    """
    cache_hits = []

    def check_match(column_spec, target_key):
        for target_col, target_width in target_key:
            search_result = list(
                filter(
                    lambda x: (x[0] == target_col and target_width % x[1] == 0),
                    column_spec,
                )
            )
            if not search_result:
                return False
        return True

    for column_spec in cache:
        if check_match(column_spec, target_key):
            for entry in cache[column_spec]:
                cache_hits.append(entry)
    return cache_hits


def quantize(df: pd.DataFrame, domain: Dict[str, int]) -> pd.DataFrame:
    """Quantizes each column in ``df`` based on domain sizes stored in
        ``domain``

    Args:
        df: DataFrame to be quantized
        domain: Dictionary where the key is the name of the attribute and the
            value is the domain size.

    Returns:
        Quantized dataframe
    """
    new = df.copy()
    for col in domain:
        new[col] = pd.cut(df[col], domain[col]).cat.codes
    return new


def specification(
    noisy_answer: np.ndarray,
    true_answer: np.ndarray,
    error: np.ndarray,
    column_spec: Dict,
    domain: Dict[str, int],
    column_names: List = None,
):
    """
    Returns the visualization specification DataFrame in Altair-parse-able
    format. Uses MultiIndexing for easy insertion of vector-of-counts format,
    then collapses the hierarchical indexing.

    Args:
       noisy_answer: Array object containing private workload answers.
       true_answer: Array object containing true workload answers.
       error: Array object containing error
       column_spec: Dict with column names as keys and either bin widths
          or actual bin ranges as values.

    Returns:
       DataFrame holding the specification.

       For example, for a visualization consisting of columns 'a' and 'b' with:

       noisy_answer = np.arange(4)
       true_answer = np.ones(4)
       error = np.zeros(4)
       column_spec = {'a': 5, 'b':1}
       domain = {'a':10, 'b':2}
       specification(noisy_answer, true_answer, error, column_spec, domain)

                         noisy_count  true_count  plus_error  minus_error  error
          a       b
          (0, 5)  (0, 1)         0.0         1.0         0.0          0.0    0.0
                  (1, 2)         1.0         1.0         1.0          1.0    0.0
          (5, 10) (0, 1)         2.0         1.0         2.0          2.0    0.0
                  (1, 2)         3.0         1.0         3.0          3.0    0.0
    """
    _n = len(domain)

    if column_names is not None:
        assert len(column_names) == 5

    xlabels = []
    for attr, bin_width in column_spec.items():
        if isinstance(bin_width, int):
            size = domain[attr]
            temp = []
            for i in range(size // bin_width):
                # NOTE: quick fix which should be removed later
                if attr == "income":
                    temp.append(
                        "$"
                        + str(i * bin_width * 5)
                        + "k - "
                        + "$"
                        + str((i + 1) * bin_width * 5 - 1)
                        + "k"
                    )  # create a list of tuples
                elif attr == "marital":
                    temp = [
                        "Married",
                        "Absent",
                        "Separated",
                        "Widowed",
                        "Divorced",
                        "Divorced/Widowed",
                        "Never Married",
                    ]
                elif attr == "race":
                    temp = ["White", "Black", "American Indian", "Chinese"]
                elif attr == "age":
                    temp.append(
                        str(i * bin_width + 15)
                        + "-"
                        + str(((i + 1) * bin_width) - 1 + 15)
                    )  # create a list of tuples
                elif attr == "Incident Year":
                    temp = [str(x) for x in range(1990, 2005)]
                elif attr == "Operator":
                    temp = [
                        "American",
                        "Delta",
                        "Fedex",
                        "Military",
                        "Southwest",
                        "United",
                    ]
                elif attr == "Species Name":
                    temp = [
                        "AMERICAN KESTREL",
                        "BARN SWALLOW",
                        "COYOTE",
                        "GULL",
                        "HORNED LARK",
                        "MOURNING DOVE",
                        "PIGEON",
                    ]
                elif attr == "Incident Month":
                    temp = [
                        "JAN.",
                        "FEB.",
                        "MAR.",
                        "APR.",
                        "MAY",
                        "JUN.",
                        "JUL.",
                        "AUG.",
                        "SEP.",
                        "OCT.",
                        "NOV.",
                        "DEC.",
                    ]
                elif attr == "Flight Phase":
                    temp = [
                        "APPROACH",
                        "TAKEOFF RUN",
                        "CLIMB",
                        "LANDING ROLL",
                        "DESCENT",
                        "EN ROUTE",
                        "TAXI",
                        "ARRIVAL",
                        "PARKED",
                        "DEPARTURE",
                    ]
                elif attr == "Visibility":
                    temp = ["DAY", "NIGHT", "DUSK", "DAWN", "UNKNOWN"]
                else:
                    pass
            xlabels.append(temp)
        elif isinstance(bin_width, List):
            to_str = [str(x) for x in bin_width]
            xlabels.append(to_str)  # JOIE: should check if bins are legit
        else:
            raise TypeError("Every entry in column_spec should be of type List or str")
    if _n > 1:
        index = pd.MultiIndex.from_product(xlabels, names=column_spec.keys())
    else:
        index = pd.Index(xlabels)

    values = np.asarray(
        [
            np.round(noisy_answer, decimals=1),
            true_answer,
            np.round(noisy_answer + error, decimals=1),
            np.round(noisy_answer - error, decimals=1),
            np.round(error, decimals=1),
        ]
    ).T

    if column_names is None:
        column_names = [
            "noisy_count",
            "true_count",
            "plus_error",
            "minus_error",
            "error",
        ]
    specification = pd.DataFrame(
        index=index,
        data=values,
        columns=column_names,
    )
    return specification


def remeasure(
    W: builder.Workload, measurements: List[CacheEntry], error_type="L2"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    If the workload requested has no previously cached answers, return the
    answers directly. If there are cached answers, combine them using inverse
    variance weighting and then return.

    Args:
       W: Workload requested for measurements
       measurements: A list where each entry is a CacheEntry which holds the
          workload matrix, strategy matrix, and epsilon spent of previous
          measurements.
    Returns:
       A tuple where the first entry is the noisy answers and the second entry
       is the closed form error.
    """
    if len(measurements) == 1:
        err = error.per_query_error(
            W=W.matrix, A=measurements[0].strategy, eps=measurements[0].eps
        )
        answer = W.matrix @ measurements[0].At_y
        return answer, err

    # JOIE: perhaps it's better to use a different storage structure
    #       which will allow me to select by column
    strategies = np.asarray([m.strategy for m in measurements])
    epsilons = np.asarray([m.eps for m in measurements])
    answers = np.asarray([(W.matrix @ m.At_y) for m in measurements])
    answers = answers.T

    err = error.union_error(W, strategies, epsilons) #
    combined_ans, combined_err = error.inverse_variance_weighting(
        answers=answers, variances=err
    )
    return combined_ans, combined_err


"""
Automated routines for fast experiments
"""


def initialize_backend(
    domain: Dict[str, int], data_path: str, budget: float
) -> BackEnd:
    """
    Returns backend object, initialized for the CPS dataset
    """
    data = pd.read_csv(data_path)
    data = data[["age", "income", "marital", "race"]]
    data = quantize(data, {"age": 100, "income": 100, "marital": 7, "race": 4})
    cps_dataset = Dataset(data, domain)
    return BackEnd(cps_dataset, budget=budget, seed=0)


def initialize_backend_wildlife(
    domain: Dict[str, int], data_path: str, budget: float, seed: int
) -> BackEnd:
    """
    Returns backend object, initialized for the wildlife dataset
    """
    data = pd.read_csv(data_path)
    data = quantize(
        data,
        {
            "Incident Month": 12,
            "Incident Year": 15,
            "Operator": 6,
            "Species Name": 7,
            "Species Quantity": 5,
        },
    )
    dataset = Dataset(data, domain)
    return BackEnd(dataset, budget=budget, seed=seed)


def main():
    # check if code passes tests
    import doctest

    doctest.testmod()

    # change this to point to you data_path
    data_path = "~/dp/DP-viz/data/CPS/CPS.csv"
    cps_domain = Domain(attrs=("income", "age", "marital"), shape=(50, 99, 7))
    back_end = initialize_backend(cps_domain, data_path)

    linked_hist = builder.histogram_workload(
        cps_domain.config, bin_widths={"income": 5, "age": 1}
    )
    back_end.measure_hdmm(workload=linked_hist, eps=1.0)
    specification = back_end.display(linked_hist)

    linked_hist = builder.histogram_workload(
        cps_domain.config,
        bin_widths={"income": [(0, 5), (5, 10), (10, 15), (15, 50)], "age": 1},
    )

    back_end.measure_hdmm(workload=linked_hist, eps=10.0)
    specification = back_end.display(linked_hist)


if __name__ == "__main__":
    main()
