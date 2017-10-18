import os
import numpy as np
import statsmodels.api as sm
import pandas as pd

from statsmodels.tsa.arima_process import arma_generate_sample

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test_arima_model():
	arparams = np.array([.75, -.25])
	maparams = np.array([.65, .35])

	arparams = np.r_[1, -arparams]
	maparams = np.r_[1, maparams]
	nobs = 250
	y = arma_generate_sample(arparams, maparams, nobs)

	dates = sm.tsa.datetools.dates_from_range('1980m1', length=nobs)
	y = pd.Series(y, index=dates)

	# ARMA
	######
	arma_mod = sm.tsa.ARMA(y, order=(3,2))
	arma_res = arma_mod.fit(trend='nc', disp=-1)

	print(arma_res.summary())
	print(y.tail())

	# plot
	fig, ax = plt.subplots(figsize=(10,8))
	fig = arma_res.plot_predict(start='1990-06-30', end='2005-05-31', ax=ax)
	legend = ax.legend(loc='upper left')
	fig.savefig('arma.png')

	# predict
	predict_arr = arma_res.predict(start='2000-06-30', end='2000-06-30')
	print("=== Predict Array: ===")
	print(predict_arr)

	# forecast
	forecast_arr, stderr, conf_int = arma_res.forecast(steps=20)
	print("=== Forcast Array: ===")
	print(forecast_arr)

	# ARIMA
	########
	arima_mod = sm.tsa.ARIMA(y, order=(2,1,2))
	arima_res = arima_mod.fit(trend='nc', disp=-1)

	print(arima_res.summary())
	print(y.tail())

	# plot
	fig, ax = plt.subplots(figsize=(10,8))
	fig = arima_res.plot_predict(start='1990-06-30', end='2001-05-31', ax=ax)
	legend = ax.legend(loc='upper left')
	fig.savefig('arima.png')


def test_arma_model_with_int_index():
	arparams = np.array([.75, -.25])
	maparams = np.array([.65, .35])

	arparams = np.r_[1, -arparams]
	maparams = np.r_[1, maparams]
	nobs = 250
	y = arma_generate_sample(arparams, maparams, nobs)

	# dates = list(range(1, 251))
	# y = pd.Series(y, index=dates)

	# ARMA
	######
	arma_mod = sm.tsa.ARMA(y, order=(2,2))
	arma_res = arma_mod.fit(trend='nc', disp=-1)

	print(arma_res.summary())

	# plot
	print(arma_res.predict(start=245, end=252))
	print(y[245:])
	fig, ax = plt.subplots(figsize=(10,8))
	fig = arma_res.plot_predict(start=1, end=252, ax=ax)
	legend = ax.legend(loc='upper left')
	fig.savefig('arma_int.png')


def main():
	test_arima_model()
	test_arma_model_with_int_index()


if __name__ == '__main__':
	main()