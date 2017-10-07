import numpy as np


def detect_anoms(time_series, k=0.49, alpha=0.05, num_obs_per_period=None,
	             use_decomp=True, use_esd=False, one_tail=True,
                 upper_tail=True, verbose=False):
	"""
	Detects anomalies in a time series using S-H-ESD.

	Args:
		time_series: Time series to perform anomaly detection on.
		k: Maximum number of anomalies that S-H-ESD will detect as a percentage of the data.
		alpha: The level of statistical significance with which to accept or reject anomalies.
		num_obs_per_period: Defines the number of observations in a single period, and used during seasonal decomposition.
		use_decomp: Use seasonal decomposition during anomaly detection.
		use_esd: Uses regular ESD instead of hybrid-ESD. Note hybrid-ESD is more statistically robust.
		one_tail: If TRUE only positive or negative going anomalies are detected depending on if upper_tail is TRUE or FALSE.
		upper_tail: If TRUE and one_tail is also TRUE, detect only positive going (right-tailed) anomalies. If FALSE and one_tail is TRUE, only detect negative (left-tailed) anomalies.
		verbose: Additionally printing for debugging.

	Returns:
		A list containing the anomalies (anoms) and decomposition components (stl).
	"""
	pass
