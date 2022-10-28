from mbi import Domain
from ektelo import workload
from ektelo.client import selection
from ektelo.matrix import EkteloMatrix, Kronecker
from typing import NamedTuple, Dict, List, Tuple
import numpy as np


class Workload(NamedTuple):
    matrix: Kronecker
    column_spec: Dict


def histogram_workload(domain: Dict[str, int], bin_widths) -> Workload:
    """
    Args:
            domain (dict): Dictionary where key is the name of the column and value
            is the size

            bin_widths (dict): Dictionary where the key is the name of the
                    column that the histogram will be over and the value is either:
                    (1) The integer which represents the width of the bin if the histogram is uniform
                    (2) A list of bin widths if the histogram is custom
                    (3) The tuple which contains the selected index if the entry is a single predicate query.

    Returns:
            NamedTuple of Workload type holding resulting workload
    """
    kron = []
    bin_widths = dict(sorted(bin_widths.items()))
    for col, size in domain.items():
        if col in bin_widths:
            if isinstance(bin_widths[col], int):
                kron.append(uniform_hist_matrix(size, bin_widths[col]))
            elif isinstance(bin_widths[col], List):
                kron.append(custom_hist_matrix(size, bin_widths[col]))
            elif isinstance(bin_widths[col], tuple):
                kron.append(selector_matrix(size, bin_widths[col][0]))
        else:
            kron.append(workload.Total(size))
    return Workload(matrix=workload.Kronecker(kron), column_spec=bin_widths)


def uniform_hist_matrix(m: int, bin_width: int) -> EkteloMatrix:
    """Single histogram query building block if bins queries are uniform
    width. Note that if bin_width = 1, this will be the identity query.

    Args:
            m (int): Domain size of column.
            bin_width (int): width of each bin. ``m`` must be divisible by
                    ``bin_width``

    Returns:
            EkteloMatrix containing uniform histogram workload matrix with
            ``bin_width`` wide bins

    Examples:
            >>> uniform_hist_matrix(4,2)
            <2x4 EkteloMatrix with dtype=float64>

            >>> uniform_hist_matrix(4,2).matrix
            array([[1., 1., 0., 0.],
                   [0., 0., 1., 1.]])

            >>> uniform_hist_matrix(4,3)
            Traceback (most recent call last):
            ...
            AssertionError: bin_width must be divisible by m

            >>> uniform_hist_matrix(4,1).matrix
            array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])

    """
    assert m % bin_width == 0, "bin_width must be divisible by m"
    n = m // bin_width
    matrix = np.zeros((n, m))
    for i in range(0, n):
        start, end = (i * bin_width, (i + 1) * bin_width)
        matrix[i, start:end] = np.ones(bin_width)
    return EkteloMatrix(matrix)


def custom_hist_matrix(m: int, bins: List[tuple]) -> EkteloMatrix:
    """Single histogram workload building block if bins requested are not
    uniform.

    Args:
            m (int): Domain size of column.
            bins (List[tuple]): List of of tuples where each tuple defines the range
                    for a single bin. We consider the ranges to be open at the end, i.e: [a,b).

    Returns:
            EkteloMatrix holding workload matrix

    Examples:
            >>> custom_hist_matrix(5, [(0,1),(1,3),(3,5)])
            <3x5 EkteloMatrix with dtype=float64>

            >>> custom_hist_matrix(5, [(0,1),(1,3),(3,5)]).matrix
            array([[1., 0., 0., 0., 0.],
                   [0., 1., 1., 0., 0.],
                   [0., 0., 0., 1., 1.]])
    """
    # JOIE: Would be good to add some assertions to confirm that content of
    # ``bins`` is expected

    n = len(bins)
    matrix = np.zeros((n, m))
    for idx, interval in enumerate(bins):
        start, end = interval
        matrix[idx, start:end] = 1
    return EkteloMatrix(matrix)


def selector_matrix(m: int, selected_idx: int) -> EkteloMatrix:
    """Returns a matrix consisting of a single predicate query

    Args:
            m (int): Domain size of column.
            selection(int): Index of the selected attribute

    Returns:
            EkteloMatrix holding the query

    Examples:
            >>> selector_query(5, 1)
            <5x1 EkteloMatrix with dtype=float64>

            >>> custom_hist_matrix(5, 1).matrix
            array([[0., 1., 0., 0., 0.]])
    """

    matrix = np.zeros((1, m))
    matrix[:, selection] = 1
    print(matrix)
    return EkteloMatrix(matrix)
