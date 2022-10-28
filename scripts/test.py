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
import backend

CPS_PATH = "~/dp/DP-viz/data/CPS/CPS.csv"
allrange = workload.Kronecker(
    [workload.AllRange(100), workload.AllRange(99), workload.Total(7)]
)
identity = workload.Kronecker(
    [workload.Identity(100), workload.Identity(99), workload.Total(7)]
)
prefix = workload.Kronecker(
    [workload.Prefix(100), workload.Prefix(99), workload.Total(7)]
)


def main():
    cps = pd.read_csv(CPS_PATH)
    cps = cps[["income", "age", "marital"]]
    cps_domain = Domain(cps.columns, (100, 99, 7))

    m = backend.BackEnd(cps, cps_domain)
    num_trials = 3
    # 	print(m.vector_of_counts)
    # 	print(m.vector_of_counts.query('95 <= income <= 99'))
    # 	print(m.vector_of_counts.loc[50:100, 0, 1])
    # 	m.vector_of_counts.loc[50:100, 0, 1] = np.zeros((50,1))
    # 	print(m.vector_of_counts.loc[50:100])
    kron_of_hists = workload.Kronecker(
        [
            utility.histogram_matrix(100, 10),
            utility.histogram_matrix(99, 3),
            workload.Total(7),
        ]
    )
    projected_domain = cps_domain.project(["income", "age"])

    print("running hdmm")
    # m.measurements = m.run_hdmm(allrange, 3, post_proc='ls', eps=1.0)
    m.measurements = m.run_hdmm(kron_of_hists, 2, post_proc="ls", eps=1.0)
    print("done hdmm")

    # 	df_data = m.marginal_query(['income','age'], eps=1.0)
    # 	print(df_data)
    # x_d = df_data.reset_index(['income', 'age'])
    error1 = error.expected_error(W=kron_of_hists, A=m.strategy, eps=1.0)
    error2 = error.expected_error(W=kron_of_hists, A=kron_of_hists, eps=1.0)
    error3 = error.expected_error(W=kron_of_hists, A=identity, eps=1.0)
    print("Error with P-Identity: ", error1)
    print("Error with Histogram: ", error2)
    print("Error with Identity: ", error3)


# 	df_data = m.histogram_query(['income','age'], eps=1.0, bin_widths=[10, 3])
# 	print(df_data)


if __name__ == "__main__":
    main()
