import pandas as pd
import numpy as np
import sys

sys.path.append("../src")
import binning
import backend

bins = 500
step_lon = (-73.77 - -74.05) / bins
step_lat = (40.91 - 40.61) / bins


bins = {
    "Lon": np.r_[-np.inf, np.arange(-74.05, -73.77, step=step_lon), np.inf],
    "Lat": np.r_[-np.inf, np.arange(40.61, 40.91, step=step_lat), np.inf],
}


x = pd.read_csv(
    "/users/joiewu/dp/dp-viz/data/uber-tlc-foil-response-master/uber-trip-data/uber-raw-data-jul14.csv"
)
discretized = binning.discretize(x, BINS=bins)

M = backend.BackEnd(discretized, 1.0)

print("Measuring PGM....")
model = M.measure_pgm([("Lon", "Lat")], iters=1000, eps=1.0)
synthetic_data = binning.undo_discretize(model.synthetic_data(), BINS=bins)

with open("synthetic_data.csv", "wb") as f:
    synthetic_data.to_csv(f)
