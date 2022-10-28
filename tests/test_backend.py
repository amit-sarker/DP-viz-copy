import unittest
import sys
sys.path.append('../src')
import src.backend
import pandas as pd
import numpy as np
from mbi import Domain

# Joie: TODO: port these tests to test_backend_updated

class TestBackEnd(unittest.TestCase):
	def setUp(self):
		'''
			The underlying dataset looks like this:

		      a b c d
		    [[1,1,0,0]
		     [0,0,1,1]
		     [1,1,0,1]]
		'''

		CPS_PATH = '~/dp/DP-viz/data/CPS/CPS.csv'
		cps = pd.read_csv(CPS_PATH)
		self.df = cps[['income', 'age', 'marital']]
		cps_domain = Domain(self.df.columns, (100,99,7))
		self.domain = cps_domain
		self.back_end = src.backend.BackEnd(self.df, self.domain)

	def test_to_bin(self):
		df, domain, back_end = (self.df, self.domain, self.back_end)

		binned_df = back_end.to_bin(df)
		true_df_size = len(df)

		self.assertEqual(true_df_size, len(binned_df))

	def test_vectorize(self):
		df, domain, back_end = (self.df, self.domain, self.back_end)

		binned_df = back_end.to_bin(df)
		vectorized_df = back_end.vectorize(binned_df, domain)
		vector = vectorized_df.values.flatten()
		true_df_size = len(df)

		self.assertEqual(true_df_size, np.sum(vector))



if __name__ == '__main__':
	unittest.main()
