import numpy as np
import statsmodels.api as sm
from scipy.stats import t


def detect_anoms_s_h_esd(time_series, k=0.49, alpha=0.05,
                         frequency=3, strict=True):
    """
    Detects anomalies in a time series using S-H-ESD.

    Args:
        time_series: Time series to perform anomaly detection on, np.array[int].
        k: Maximum number of anomalies that S-H-ESD will detect as a percentage of the data.
        alpha: The level of statistical significance with which to accept or reject anomalies.
        frequency: Defines the number of observations in a single period, and used during seasonal decomposition.
        strict: strictly apply S-H-ESD algorithm if TRUE.

    Returns:
        A list containing the anomalies (anoms) and decomposition components (stl).
        Returns if can not find any anomaly.

    Note:
        Grubbs test allows one-sided tests (i.e., you can specify a minimum test or the maximum test) in addition
        to two-sided tests (both the minimum and the maximum value are tested). The generalized ESD test is
        restricted to two-sided tests.
    """

    raw_data = np.copy(time_series)
    n = len(raw_data)

    # Decompose Data
    data = sm.tsa.seasonal_decompose(raw_data, freq=frequency)

    # Remove the seasonal component and trend component, to get univariate variable
    uni_var = data.observed - data.trend - data.seasonal

    # Hanle nan
    m = np.nanmean(uni_var)
    for i in range(len(uni_var)):
        if np.isnan(uni_var[i]):
            uni_var[i] = m

    expected_data = data.trend + data.seasonal

    return generalized_esd_test(uni_var, k, alpha, strict)


def detect_anoms_ar_esd(time_series, k=0.49, alpha=0.05,
                        strict=True):
    """
    Detects anomalies in a time series using AR model and ESD tests.

    Args:
        time_series: Time series to perform anomaly detection on, np.array[int].
        k: Maximum number of anomalies that ESD will detect as a percentage of the data.
        alpha: The level of statistical significance with which to accept or reject anomalies.
        strict: strictly apply ESD algorithm if TRUE.

    Returns:
        A list containing the anomalies (anoms) and decomposition components (stl).
        Returns if can not find any anomaly.

    Note:
        Grubbs test allows one-sided tests (i.e., you can specify a minimum test or the maximum test) in addition
        to two-sided tests (both the minimum and the maximum value are tested). The generalized ESD test is
        restricted to two-sided tests.
    """
    # Build ARIMA Model
    N = len(time_series)
    ar = sm.tsa.AR(time_series)
    ar_model = ar.fit(trend='nc', disp=-1)
    predicted_values = ar_model.predict(start=1, end=N)
    uni_var = time_series - predicted_values

    # Perform Generalized ESD test to find anomalies
    return generalized_esd_test(uni_var, k, alpha, strict)


def detect_anoms_arima_esd(time_series, pdp_tuple, k=0.49, alpha=0.05,
                           strict=True):
    """
    Detects anomalies in a time series using ARIMA model and ESD tests.

    Args:
        time_series: Time series to perform anomaly detection on, np.array[int].
        pdp_tuple: The (p,d,q) order of the model for the number of AR parameters, differences,
                     and MA parameters to use.
        k: Maximum number of anomalies that ESD will detect as a percentage of the data.
        alpha: The level of statistical significance with which to accept or reject anomalies.
        strict: strictly apply ESD algorithm if TRUE.

    Returns:
        A list containing the anomalies (anoms) and decomposition components (stl).
        Returns if can not find any anomaly.

    Note:
        Grubbs test allows one-sided tests (i.e., you can specify a minimum test or the maximum test) in addition
        to two-sided tests (both the minimum and the maximum value are tested). The generalized ESD test is
        restricted to two-sided tests.
    """

    # Build ARIMA Model
    N = len(time_series)
    arima = sm.tsa.ARIMA(time_series, order=(2,1,2))
    arima_model = arima.fit(trend='nc', disp=-1)
    predicted_values = arima_model.predict(start=1, end=N)
    uni_var = time_series - predicted_values

    # Perform Generalized ESD test to find anomalies
    return generalized_esd_test(uni_var, k, alpha, strict)


def generalized_esd_test(var, k, alpha, strict=True):
    """
    This is an implementation of generalized ESD test.

    Args:
        var: Vars to perform generalized esd test.
        k: Maximum number of anomalies that ESD will detect as a percentage of the data.
        alpha: The level of statistical significance with which to accept or reject anomalies.
        strict: strictly apply ESD algorithm if TRUE.
    """
    n = len(var)
    max_outliers = int(n*k)
    anoms_index = [None] * max_outliers
    num_anoms = 0

    for i in range(max_outliers):

        # mean and standard deviation
        mean = np.nanmean(var)
        std = np.nanstd(var)

        # In Case, this series is constant, let's break
        if std == 0:
            break

        residuals = np.abs((var - mean)/std)

        # Find residual with max z-values
        max_r_index = np.argmax(residuals)
        max_r = residuals[max_r_index]
        anoms_index[i] = max_r_index

        # Calcualate Critical Value
        # critical_value = ((n-1)/np.sqrt(n))*np.sqrt(np.power(t.ppf(alpha/(2*n),n-2),2)/(n-2+np.power(t.ppf(alpha/(2*n),n-2),2)))

        p = 1 - alpha/(2*(n-(i+1)+1))
        tpv = t.ppf(p, n-(i+1)-1)

        critical_value = tpv*(n-(i+1))/np.sqrt((n-(i+1)-1+np.power(tpv,2))*(n-(i+1)+1))

        if max_r > critical_value:
            num_anoms = i + 1
        elif not strict:
            # Let's break if we don't strict apply esd.
            break

        # Do not delete current selected item from numpy array, it's very expensive.
        # Let's input the new mean for next iteration, then this value should have no
        # contribution to next iteration.
        var[max_r_index] = (mean * n - var[max_r_index])/(n - 1)

    return anoms_index[:num_anoms] if num_anoms > 0 else None
