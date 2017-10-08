import numpy as np
import pandas as pd
import statsmodels.api as sm


def detect_anoms_esd(time_series, k=0.49, alpha=0.05,
					 frequency=None, one_tail=True,
                     upper_tail=True, verbose=False):
	"""
	Detects anomalies in a time series using S-H-ESD.

	Args:
		time_series: Time series to perform anomaly detection on, np.array[int].
		k: Maximum number of anomalies that S-H-ESD will detect as a percentage of the data.
		alpha: The level of statistical significance with which to accept or reject anomalies.
		frequency: Defines the number of observations in a single period, and used during seasonal decomposition.
		one_tail: If TRUE only positive or negative going anomalies are detected depending on if upper_tail is TRUE or FALSE.
		upper_tail: If TRUE and one_tail is also TRUE, detect only positive going (right-tailed) anomalies. If FALSE and one_tail is TRUE, only detect negative (left-tailed) anomalies.
		verbose: Additionally printing for debugging.

	Returns:
		A list containing the anomalies (anoms) and decomposition components (stl).
	"""

	raw_data = np.copy(time_series)
	num_obs = len(raw_data)

	# Decompose Data
	data = sm.tsa.seasonal_decompose(raw_data, freq=frequency)

	expected_data = data.trend + data.seasonal

	# Maximum number of outliers that S-H-ESD can detect (e.g. 49% of data)
    max_outliers = int(num_obs*k)


