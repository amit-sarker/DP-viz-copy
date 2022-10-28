import sys

sys.path.append("../src")
import utility
import pandas as pd
import numpy as np
import functools
import backend
import ektelo
from ektelo import workload
from ektelo import matrix as ektelo_matrix
from ektelo.matrix import EkteloMatrix
from mbi import Domain
import altair as alt
from hdmm import templates
from hdmm import error as hdmm_error
from utility import histogram_matrix

CPS_PATH = "~/dp/DP-viz/data/CPS/CPS.csv"


def get_workload_get_strategy(df, domain, eps1, eps2, remeasure="replace"):
    M = backend.BackEnd(df, domain)

    identity_workload = workload.Kronecker(
        [workload.Identity(50), workload.Identity(99), workload.Total(7)]
    )

    # first measurement
    print("First measurement....")
    ans1, strategy1, x_est1 = M.hdmm_run(workload=identity_workload, eps=eps1)

    print("Performing remeasurement")
    range_workload = histogram_matrix(50, 5)
    range_workload.matrix[6:] = 0

    remeasure_workload = workload.Kronecker(
        [range_workload, workload.Identity(99), workload.Total(7)]
    )
    histogram_workload = workload.Kronecker(
        [histogram_matrix(50, 5), workload.Identity(99), workload.Total(7)]
    )

    histogram_labels = {"age": np.arange(0, 50, 5), "income": np.arange(99)}

    if remeasure == "disjoint":
        y2, strategy2, x_est2 = M.hdmm_run(workload=remeasure_workload, eps=eps2)
    if remeasure == "replace":
        y2, strategy2, x_est2 = M.hdmm_run(workload=histogram_workload, eps=eps2)

    return (
        histogram_workload,
        [strategy1, strategy2],
        np.hstack(
            [
                (histogram_workload @ x_est1)[:, np.newaxis],
                (histogram_workload @ x_est2)[:, np.newaxis],
            ]
        ),
        [x_est1, x_est2],
    )


def kron_strategy_error(W, A, eps):
    """
    params:
    W (single kron)
    A (single kron)
    Takes the error vector of a kron strategy and a kron workload
    Vector consists of entries || wA^+ || for each row 'w' in matrix W.
    """
    # vector_norms = [ EkteloMatrix(np.linalg.norm(( Wi.matrix @ Ai.pinv().matrix ).T, axis=0)**2) for Wi, Ai in zip(W.matrices, A.matrices) ]
    # vector_norms = [ error.per_query_error(Wi, Ai, eps=1.0)[:,np.newaxis] for Wi, Ai in zip(W.matrices, A.matrices) ]
    vector_norms = hdmm_error.per_query_error(W, A, eps)
    # return functools.reduce(np.kron, vector_norms).flatten()
    return vector_norms
    # return ektelo_matrix.Kronecker(vector_norms).matrix


def union_error(kron_W, A, epsilons):
    """
    params:
    kron_W: A single kron
    A (list of strategie(s))
    """
    num_queries = kron_W.shape[0]
    err = [kron_strategy_error(kron_W, kron_A, eps) for kron_A, eps in zip(A, epsilons)]
    err = np.asarray(err)
    err = err.reshape((num_queries, -1))
    return err


def inverse_variance_weighting(error, observations):
    """
    Refer to Wikipedia page of inverse variance weighting for formula

    Error must be of shape (num_queries, num_strategies) where num_strategies is
    the number of strategies in the union of krons and num_queries is the sum of
    rows in the union of strategies

    NOTE: We assume that all strategies support all workloads in this function,
    this may not be the case for other use cases
    """
    s = np.sum(observations / error, axis=1)  # numerator of a_hat
    var_a_hat = 1 / np.sum(1 / error, axis=1)  # denominator of a_hat
    a_hat = s * var_a_hat
    return a_hat, var_a_hat


def stacked_least_squares(A_1, A_2, y_1, y_2):
    """
    Wildly inefficient, do not use for non trivial strategy matrics
    """
    y = np.hstack([y_1, y_2])
    A = matrix.VStack([A_1, A_2])
    A_inv = A.pinv()
    return A_inv @ y


def replace(eps1, eps2):
    cps = pd.read_csv(CPS_PATH)
    cps_df = cps[["income", "age", "marital"]]
    cps_domain = Domain(cps_df.columns, (50, 99, 7))
    eps1 = eps1
    eps2 = eps2
    hist_workload, strategies, observations, x_est = get_workload_get_strategy(
        cps_df, cps_domain, eps1, eps2, remeasure="replace"
    )

    back_end = backend.BackEnd(cps_df, cps_domain)
    identity = workload.Kronecker(
        [workload.Identity(50), workload.Identity(99), workload.Total(7)]
    )

    """ UNIT TESTS Start """
    assert (
        kron_strategy_error(hist_workload, strategies[0], eps1).shape[0]
        == hist_workload.shape[0]
    )
    err = union_error(hist_workload, strategies, [eps1, eps2])

    assert err.shape[0] == hist_workload.shape[0], "Shapes are {0} and {1}".format(
        union_error(hist_workload, strategies).shape, workload.shape
    )

    assert (
        observations.shape[0] == hist_workload.shape[0]
    ), "Shapes are {0} and {1}".format(observations.shape[0], hist_workload.shape[0])

    estimates, error = inverse_variance_weighting(error=err, observations=observations)
    assert (
        estimates.shape[0] == hist_workload.shape[0]
    ), "Shapes are {0} and {1} respectively".format(
        estimates.shape[0], hist_workload.shape[0]
    )
    """ END """

    histogram_labels = {"income": np.arange(0, 50, 5), "age": np.arange(99)}
    index = pd.MultiIndex.from_product(
        histogram_labels.values(), names=histogram_labels.keys()
    )
    true_count = hist_workload @ back_end.vector
    values = np.asarray(
        [estimates, true_count, estimates + error, estimates - error, error]
    ).T
    specification_remeasured = pd.DataFrame(
        index=index,
        data=values,
        columns=["noisy_count", "true_count", "plus_error", "minus_error", "error"],
    )

    specification_1 = back_end.viz_specification(
        workload=hist_workload,
        col_labels=histogram_labels,
        synthetic_data=x_est[0],
        strategy=strategies[0],
        eps=1.0,
    )
    return specification_1, specification_remeasured, back_end.vector_df


def disjoint():
    cps = pd.read_csv(CPS_PATH)
    cps_df = cps[["income", "age", "marital"]]
    cps_domain = Domain(cps_df.columns, (50, 99, 7))
    eps1 = 1.0
    eps2 = 0.5
    hist_workload, strategies, observations, x_est = get_workload_get_strategy(
        cps_df, cps_domain, eps1, eps2, remeasure="disjoint"
    )

    back_end = backend.BackEnd(cps_df, cps_domain)
    identity = workload.Kronecker(
        [workload.Identity(50), workload.Identity(99), workload.Total(7)]
    )

    """ UNIT TESTS Start """
    assert (
        kron_strategy_error(hist_workload, strategies[0], eps1).shape[0]
        == hist_workload.shape[0]
    )
    err = union_error(hist_workload, strategies, [eps1, eps2])

    assert err.shape[0] == hist_workload.shape[0], "Shapes are {0} and {1}".format(
        union_error(hist_workload, strategies).shape, workload.shape
    )

    assert (
        observations.shape[0] == hist_workload.shape[0]
    ), "Shapes are {0} and {1}".format(observations.shape[0], hist_workload.shape[0])

    estimates, error = inverse_variance_weighting(error=err, observations=observations)
    assert (
        estimates.shape[0] == hist_workload.shape[0]
    ), "Shapes are {0} and {1} respectively".format(
        estimates.shape[0], hist_workload.shape[0]
    )
    """ END """

    histogram_labels = {"income": np.arange(0, 50, 5), "age": np.arange(99)}
    index = pd.MultiIndex.from_product(
        histogram_labels.values(), names=histogram_labels.keys()
    )
    true_count = hist_workload @ back_end.vector
    values = np.asarray(
        [estimates, true_count, estimates + error, estimates - error, error]
    ).T
    specification_remeasured = pd.DataFrame(
        index=index,
        data=values,
        columns=["noisy_count", "true_count", "plus_error", "minus_error", "error"],
    )

    specification_1 = back_end.viz_specification(
        workload=hist_workload,
        col_labels=histogram_labels,
        synthetic_data=x_est[0],
        strategy=strategies[0],
        eps=1.0,
    )
    specification_remeasured[
        np.arange(30, 50, 5), ["noisy_count", "error", "plus_error", "minus_error"]
    ] = specification_1.loc[np.arange(30, 50, 5), ["noisy_count", "error"]]
    return specification_1, specification_remeasured, back_end.vector_df


if __name__ == "__main__":
    error_remeasure = []
    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for eps2 in epsilons:
        specification_1, specification_remeasured, _ = replace(eps1=1.0, eps2=eps2)
        error_remeasure.append(specification_remeasured.error.sum())
    error_first = specification
