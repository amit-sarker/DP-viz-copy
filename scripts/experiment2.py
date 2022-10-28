import sys
sys.path.append('../src')
from typing import NamedTuple, Tuple, Dict, List
from collections import defaultdict
from ektelo import workload, matrix
from hdmm import inference
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

# Local imports
from mbi import Domain, Dataset, FactoredInference
import workload_builder as builder
import mechanisms
import error
import backend

def relative_error(true_count, noisy_count, delta):
	denominator = list(map(lambda i: max(i,delta), true_count))
	return (np.abs(true_count-noisy_count)/denominator)

if __name__ == '__main__':
	# Load Data	
	data_path = "~/dp/DP-viz/data/CPS/CPS.csv"
	cps_domain = Domain(attrs=("income", "age", "marital"), shape=(100, 99, 7))
	budget = 10.0
	back_end_VeDA = backend.initialize_backend(cps_domain, data_path,
		budget=budget)

	final_workload = builder.histogram_workload(
						cps_domain.config, bin_widths={"income": 5, "age": 3}
					)

	id_workload = builder.histogram_workload(
						cps_domain.config, bin_widths={"income": 1, "age": 1}
					)

	back_end_VeDA.measure_hdmm(workload=id_workload, eps=1.0)

	# display final visualization
	specification_VeDA = back_end_VeDA.display(final_workload)

	specification_VeDA.to_csv('../dataframes/workload.csv')
	
