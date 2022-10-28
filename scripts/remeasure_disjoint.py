import sys

sys.path.append("../src")
import utility
import pandas as pd
import numpy as np

import backend
from ektelo import workload, matrix
from ektelo.matrix import EkteloMatrix
from mbi import Domain
import altair as alt
from hdmm import error, templates
from utility import histogram_matrix

"""
Example: Backend first measures a two-way Identity query over age and income, 
then it accepts a remeasurement over a certain range of the income domain,
updating only the x_hat entries that it remeasures.
"""

CPS_PATH = "~/dp/DP-viz/data/CPS/CPS.csv"


def main():
    cps = pd.read_csv(CPS_PATH)
    cps_df = cps[["income", "age", "marital"]]
    cps_domain = Domain(cps_df.columns, (50, 99, 7))
    eps1 = 1.0
    eps2 = 1.0

    back_end = backend.BackEnd(cps_df, cps_domain)

    identity = workload.Kronecker(
        [workload.Identity(50), workload.Identity(99), workload.Total(7)]
    )

    region = (1, 5)  # selected region of interest (brush over box on histogram)
    bin_width = 5
    range_query = histogram_matrix(50, bin_width)
    range_query.matrix[: region[0]] = 0
    range_query.matrix[region[1] :] = 0

    remeasure_query = workload.Kronecker(
        [range_query, workload.Identity(99), workload.Total(7)]
    )
    histogram_query = workload.Kronecker(
        [histogram_matrix(50, bin_width), workload.Identity(99), workload.Total(7)]
    )

    identity_labels = {"age": np.arange(50), "income": np.arange(99)}
    histogram_labels = {"age": np.arange(0, 50, 5), "income": np.arange(99)}

    xhat1, strategy1, specification1 = back_end.measure_workload(
        workload=identity, col_labels=identity_labels, eps=eps1
    )

    specification2 = back_end.viz_specification(
        workload=histogram_query,
        synthetic_data=xhat1,
        col_labels=histogram_labels,
        strategy=strategy1,
        eps=eps1,
    )

    xhat2, strategy2, specification3 = back_end.measure_workload(
        workload=histogram_query,
        eps=eps2,
        col_labels=histogram_labels,
        remeasure_workload=remeasure_query,
    )

    specification2.loc[np.arange(5, 25, 5), ["count", "error"]] = specification3.loc[
        np.arange(5, 25, 5), ["count", "error"]
    ]

    return specification2, specification3


if __name__ == "__main__":
    specification1, specification2 = main()
    print(specification1)
