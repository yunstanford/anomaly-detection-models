import numpy as np


def correlate_metrics(time_series_matrix, target_ts_index, correlation_level=0.9):
	"""
	Correlate target metrics among a bunch of metrics.

    Args:
        time_series_matrix: time_series metrics.
		target_ts_index: index of target time series data.
		correlation_level: correlation level.

        Perf issues ?
	"""
	sz = len(time_series_matrix)
	if sz < 2:
		raise ValueError("Input Matrix should Larger than 2...")
	if target_ts_index >= sz:
		raise ValueError("Target Out of Index...")

	correlation_table = _calulate_correlation_table(time_series_matrix)
	print(correlation_table)
	correlated_index = _find_correlated_ts(correlation_table, target_ts_index, correlation_level)
	return correlated_index


def _find_correlated_ts(correlation_table, target_index, correlation_level):
	sz = len(correlation_table)
	print(sz)
	results = []
	# search column
	for i in range(0, target_index):
		if correlation_table[i][target_index] >= correlation_level:
			results.append(i)
	# search row
	for j in range(target_index + 1, sz):
		if correlation_table[target_index][j] >= correlation_level:
			results.append(j)
	return results


def _calulate_correlation_table(time_series_matrix):
	"""
	Calculate correlation table based on the time_series metrics.

    Args:
        time_series_matrix: time_series metrics
	"""
	sz = len(time_series_matrix)
	corr_table = np.zeros(shape=(sz, sz))
	for i in range(sz):
		for j in range(sz):
			if i < j:
				corr_table[i][j] = _normalized_cross_correlation(time_series_matrix[i], time_series_matrix[j])
	return corr_table


def _normalized_cross_correlation(ts1, ts2, option="full"):
	norm_ts1 = np.divide((ts1 - np.mean(ts1)), np.std(ts1) * len(ts1))
	norm_ts2 = np.divide((ts2 - np.mean(ts2)), np.std(ts2))
	return np.correlate(norm_ts1, norm_ts2)[0]


def test():
	time_series_matrix = np.array([
			[1, 2, 3],
			[3, 4, 5],
			[2, 4, 6],
			[0.5, 0.1, 0.5],
			[1, 2, 1],
			[3, 2, 1]
		])
	print(correlate_metrics(time_series_matrix, 0))

test()
