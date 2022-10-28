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

CPS_PATH = "~/dp/DP-viz/data/CPS/CPS.csv"


def main():
    cps = pd.read_csv(CPS_PATH)
    cps = cps[["income", "age", "marital"]]
    cps_domain = Domain(cps.columns, (100, 99, 7))

    M = backend.BackEnd(cps, cps_domain)
    num_trials = 3
    marginals = [("income", "age"), ("age", "marital")]

    model = M.run_pgm(marginals=marginals, iters=1, eps=1.0)
    synthetic_data = model.synthetic_data().df()


if __name__ == "__main__":
    main()
