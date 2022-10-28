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
	cps_domain = Domain(attrs=("income", "age", "marital"), shape=(50, 100, 7))

	for budget in [1.0,2.0,5.0,10.0,50.0]:
		back_end_VeDA = backend.initialize_backend(cps_domain, data_path,
			budget=budget)
		back_end_ORACLE = backend.initialize_backend(cps_domain, data_path,
			budget=budget)

		back_end_ID = backend.initialize_backend(cps_domain, data_path,
			budget=budget)


		id_workload = builder.histogram_workload(
							cps_domain.config, bin_widths={"income": 1, "age": 1}
						)
		# Measure
		pcnt = .1
		back_end_VeDA.measure_hdmm(workload=id_workload, eps=budget*pcnt,
			restarts=100)

		for i in [5,10]:
			for j in [5,10]:
				linked_hist = builder.histogram_workload(
					cps_domain.config, bin_widths={"income": i, "age": j}
				)
				back_end_VeDA.measure_hdmm(workload=linked_hist, eps=(
					(budget - budget*pcnt)/4),
					restarts=100)

		final_workload = builder.histogram_workload(
							cps_domain.config, bin_widths={"income": 10, "age": 5}
						)


		back_end_ORACLE.measure_hdmm(workload=final_workload, eps=budget,
			restarts=100)
		back_end_ID.measure_hdmm(workload=id_workload, eps=budget, restarts=100)

		# display final visualization
		specification_VeDA = back_end_VeDA.display(final_workload)
		specification_ORACLE = back_end_ORACLE.display(final_workload)
		specification_ID = back_end_ID.display(final_workload)

		print('Oracle, total error: {0}'.format(specification_ORACLE.error.sum()))
		print('Identity, total error: {0}'.format(specification_ID.error.sum()))
		print('VeDA, total error: {0}'.format(specification_VeDA.error.sum()))
		

		rel_error_veda = relative_error(specification_VeDA.true_count.to_numpy(),
			specification_VeDA.noisy_count.to_numpy(), delta=1)

		rel_error_oracle = relative_error(specification_ORACLE.true_count.to_numpy
			(), specification_ORACLE.noisy_count.to_numpy(), delta=1)

		rel_error_ID = relative_error(specification_ID.true_count.to_numpy
			(), specification_ID.noisy_count.to_numpy(), delta=1)
		print('----------------------------')
		print('Oracle, relative error: {0}'.format(np.sum(rel_error_oracle)))
		print('Identity, relative error: {0}'.format(np.sum(rel_error_ID)))
		print('VeDA, relative error: {0}'.format(np.sum(rel_error_veda)))

