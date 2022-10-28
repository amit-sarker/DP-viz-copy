import numpy as np
import pandas as pd
import ipywidgets as widgets
import sys
sys.path.append('../src')
import backend
import workload_builder as builder
from mbi import Domain, Dataset
import plots
import altair as alt
import ipdb

def main():
	data_path = '~/dp/DP-viz/data/CPS/CPS.csv'
	cps_domain = Domain(attrs=('age','income','marital'), shape=(100,100,7))
	back_end1 = backend.initialize_backend(cps_domain, data_path, budget=100.0)
	print(back_end1.dataset.datavector())


	cps_domain = Domain(attrs=('age','income','marital'), shape=(100,100,7))
	back_end2 = backend.initialize_backend(cps_domain, data_path, budget=100.0)
	print(np.all(back_end1.dataset.datavector() == back_end2.dataset.datavector()))

	#ipdb.set_trace()
#	ipdb.runcall(back_end.measure_hdmm, hist, 1.0, 20)
#	hist1 = builder.histogram_workload(cps_domain.config, bin_widths=
#		{'income':1, 'age':1, 'marital':1})
#	hist2 = builder.histogram_workload(cps_domain.config, bin_widths={'marital':1, 'age':1})
#	hist3 = builder.histogram_workload(cps_domain.config, bin_widths={'marital':1, 'income':1})
#	hist3 = builder.histogram_workload(cps_domain.config, bin_widths=
#		{'marital':1, 'income':5})
#	back_end.measure_hdmm(hist1, eps=0.1, restarts=25)
	#back_end.measure_hdmm(hist2, eps=0.1, restarts=25)
	#back_end.measure_hdmm(hist3, eps=0.1, restarts=25)

#	back_end.display(hist2)

if __name__ == '__main__':
	main()
