import numpy as np
from ektelo.matrix import EkteloMatrix, VStack, Kronecker, Weighted
from ektelo import workload
from typing import NamedTuple, Tuple, Dict, List

""" Functions below from hdmm.error, written by Ryan McKenna """


def convert_implicit(A):
    """Converts matrix ``A`` to EkteloMatrix."""
    if isinstance(A, EkteloMatrix) or isinstance(A, workload.ExplicitGram):
        return A
    return EkteloMatrix(A)


def per_query_error(W, A, eps=np.sqrt(2), delta=0, normalize=True):
    """Takes the error vector of a kron strategy and a kron workload.
    Vector consists of entries (2/eps**2)*|| wA^+ || for each row 'w' in matrix W.
    RMSE by default (normalize=True).

    Args:
        W (EkteloMatrix): Workload matrix.
        A (EkteloMatrix): Strategy matrix for W.
        eps (float, optional): Privacy parameter epsilon for epsilon-DP.
        delta (float, optional): Privacy parameter delta for approximate DP. Set
            to `0` for pure DP.
        normalize (boolean): ``True`` if RMSE, ``False`` for MSE.
    """
    W, A = convert_implicit(W), convert_implicit(A)
    if isinstance(W, VStack):
        return np.concatenate(
            [per_query_error(Q, A, eps, delta, normalize) for Q in W.matrices]
        )
    delta = A.sensitivity()
    var = 2.0 / eps ** 2
    AtA1 = A.gram().pinv()
    X = W @ AtA1 @ W.T
    err = X.diag()
    answer = var * delta ** 2 * err
    return np.sqrt(answer) if normalize else answer


""" Functions below are mine """


def query_ci(W, A, eps=np.sqrt(2)):
    """Calculates the widths of the 95% confidence interval for a workload,
    assuming that the noise is Gaussian.

    This assumption holds under the Central-Limit-Theorem. Tested in
    simulation for N = 50.

    Args:
        W (EkteloMatrix): Workload matrix.
        A (EkteloMatrix): Strategy matrix for W.
        eps (float, optional): Privacy parameter epsilon.

    Returns:
        ndarray of widths for the 95% confidence interval, one for each query
        (row) of the Workload matrix.
    """
    A1 = A.pinv()
    coeff = W @ A1  # matrix of coefficients
    sigma = np.sqrt(np.sum(2 * coeff ** 2), axis=1)  # vector of standard deviations
    ci_width = 1.96 * sigma
    return ci_width


def union_error(W, strategies, epsilons: List[float]) -> np.ndarray:
    """Calculates the error of W using each strategy matrix in ``strategies``.
        Used to obtain variances for each noisy answer to perform inverse
        variance weighting.
    Args:
        W (EkteloMatrix): A single Kronecker workload.
        strategies [EkteloMatrix]: A list of strategies, each may be a Kronecker.
            epsilons: List of epsilons spent to answer `W` using each strategy.

    Returns:
        ndarray of errors calculated for each query (row) of W.

    Example:
        Suppose
        A1 = W = [[1,1,0,0],
                [0,0,1,1]]

        A2 = I_4

        >>> W = workload.Kronecker([workload.Identity(2), workload.Total(2)])
        >>> A1 = W
        >>> A2 = workload.Kronecker([workload.Identity(2), workload.Identity(2)])
        >>> strategies = [A1, A2]
        >>> epsilons = [np.sqrt(2), np.sqrt(2)]
        >>> union_error(W, strategies, epsilons)
        array([[1., 2.],
                   [1., 2.]])
    """
    assert len(strategies) == len(epsilons)

    num_queries = W.matrix.shape[0]
    err = [per_query_error(W.matrix, A, eps) for A, eps in zip(strategies, epsilons)]
    err = np.asarray(err).T
    return err


def inverse_variance_weighting(
    answers: np.ndarray, variances: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs inverse variance weighting given a set of observations
    (``answers``) and their corresponding ``variances``. Refer to Wikipedia
    page of inverse variance weighting for more details.

    NOTE: We are not squaring ``variances`` since ``variances`` are sigma_i^2.

    Error must be of shape (num_queries, num_strategies) where num_strategies is
    the number of strategies in the union of krons and num_queries is the sum of
    rows in the union of strategies.

    NOTE: We assume that all strategies support all workloads in this function,
    this may not be the case for other use cases

    Args:
        answers: Array of observations for a given query.
        variances:  Variance of each observation.

    Returns:
        Tuple where ...
            - first entry is the combined observations obtained through
                inverse weighting.
            - second entry is the variance of the combined observations.

    Examples:
        >>> answers = np.asarray( [[10,12],[20,16]] )
        >>> variances = np.asarray( [[1,1], [2,2]] )
        >>> inverse_variance_weighting(answers, variances)
        (array([11., 18.]), array([0.5, 1. ]))

    """
    s = np.sum(answers / variances, axis=1)  # numerator of a_hat
    var_a_hat = 1 / np.sum(1 / variances, axis=1)  # denominator of a_hat
    a_hat = s * var_a_hat
    return a_hat, var_a_hat
