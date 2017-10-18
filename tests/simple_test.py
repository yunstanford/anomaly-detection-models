from anomaly_detection_models.detectors.detector import detect_anoms_s_h_esd, detect_anoms_arima_esd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample


def test_s_h_esd_simple():
	arparams = np.array([.75, -.25])
	maparams = np.array([.65, .35])

	arparams = np.r_[1, -arparams]
	maparams = np.r_[1, maparams]
	nobs = 250
	y = arma_generate_sample(arparams, maparams, nobs)
	# Find anomalies
	anomalies = detect_anoms_s_h_esd(y)
	print("Expected Anomalies: None")
	print("Acutally Found {}".format(str(anomalies)))

	print(y[100]); print(y[200])

	anom_indexes = [100, 200]
	for i in anom_indexes:
		y[i] = y[i] + 3

	anomalies = detect_anoms_s_h_esd(y, alpha=0.025)
	print("Expected Anomalies: {}".format(str(anom_indexes)))
	print("Acutally Found: {}".format(str(anomalies)))


def test_arima_esd_simple():
	arparams = np.array([.75, -.25])
	maparams = np.array([.65, .35])

	arparams = np.r_[1, -arparams]
	maparams = np.r_[1, maparams]
	nobs = 250
	y = arma_generate_sample(arparams, maparams, nobs)

	# Find anomalies
	anomalies = detect_anoms_arima_esd(y, (2,1,2))
	print("Expected Anomalies: None")
	print("Acutally Found {}".format(str(anomalies)))

	print(y[100]); print(y[200])

	anom_indexes = [100, 200]
	for i in anom_indexes:
		y[i] = y[i] + 7

	anomalies = detect_anoms_arima_esd(y, (2,1,2), alpha=0.025)
	print("Expected Anomalies: {}".format(str(anom_indexes)))
	print("Acutally Found: {}".format(str(anomalies)))


def main():
	print("===== S-H-ESD =====")
	test_s_h_esd_simple()

	print("===== ARIMA =====")
	test_arima_esd_simple()


if __name__ == '__main__':
	main()