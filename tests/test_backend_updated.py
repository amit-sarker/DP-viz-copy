import unittest
import sys
sys.path.append('../src')
import utility as utils
import backend
import pandas as pd
import numpy as np
from ektelo import workload,matrix
from mbi import Domain,Dataset

import workload_builder as builder


class TestBackEnd(unittest.TestCase):
	def setUp(self):
		'''
		Creates a dataset for testing. The underlying dataset looks like this:

		  a b c d
		[[1,1,0,0]
		 [10,0,1,1]
		 [20,1,0,1]]
		'''
		values = np.asarray([[1,1,0,0],[10,0,1,1], [20,1,0,1]])
		df = pd.DataFrame(data=values, columns=['a', 'b', 'c', 'd'])
		self.domain = Domain(attrs=['a','b','c','d'], shape=(20,2,2,2))
		self.dataset = Dataset(df=df, domain=self.domain)


	def test_histogram_workload(self):
		hist = builder.histogram_workload(domain=self.domain.config, bin_widths={'a':5})

		self.assertEqual(hist.matrix.shape, (4,160)) #20*2*2*2
		self.assertIsInstance(hist.matrix.matrices[1], matrix.Ones) # check if matrix over 'b' is a total matrix
		self.assertIsInstance(hist.matrix.matrices[2], matrix.Ones)
		self.assertIsInstance(hist.matrix.matrices[3], matrix.Ones)

	def test_measure_hdmm(self):
		back_end = backend.BackEnd(dataset=self.dataset, budget=1.0)
		hist = builder.histogram_workload(domain=self.domain.config, bin_widths={'a': 5})
		back_end.measure_hdmm(workload=hist, eps=1.0)
		self.assertEqual(list(back_end.cache.keys()), [('a',)])

		back_end.measure_hdmm(workload=hist, eps=1.0)
		self.assertEqual(len(back_end.cache[('a',)]), 2)






