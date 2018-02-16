from anomaly_detection_models.detectors.detector import detect_anoms_s_h_esd, detect_anoms_arima_esd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
import asyncio
import aiohttp


loop = asyncio.get_event_loop()


async def fetch_remote(graphite_host, graphite_port, metric, frm):
    url = "http://{host}:{port}/render?target={metric}&from={frm}&format=json".format(
            host=graphite_host,
            port=graphite_port,
            metric=metric,
            frm=frm,
        )

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()


async def get_time_series(host, port, metric, frm):
    metric_resps = await fetch_remote(host, port, metric, frm)
    metric_data = metric_resps[0]["datapoints"]
    value_list = []
    ts_list = []
    for v, t in metric_data:
        if v:
            ts_list.append(t)
            value_list.append(v)
    return np.array(value_list, dtype=np.float), np.array(ts_list, dtype=np.int)


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

    anom_indexes_1 = [100, 200]
    for i in anom_indexes_1:
        y[i] = y[i] + 5

    anom_indexes_2 = [120, 220]
    for i in anom_indexes_2:
        y[i] = y[i] - 5

    anomalies = detect_anoms_s_h_esd(y, alpha=0.025)
    print("Expected Anomalies: {}".format(str(anom_indexes_1)))
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


def test_s_h_esd_remote_ratio():
    # Fetch time series
    vals, ts = loop.run_until_complete(
        get_time_series("graphite.del.zillow.local", 80, "divideSeries(sumSeries(zdc.trends.production.pre.*.rum.homedetails-forsale.contactformrender.mobile.turnstile.out-{delighted,satisfied}),sumSeries(zdc.trends.production.pre.*.rum.homedetails-forsale.contactformrender.mobile.turnstile.in))", "-3d")
    )
    anomalies = detect_anoms_s_h_esd(vals, alpha=0.025, frequency=1440)

    if anomalies:
        print("Acutally Found: {}".format(str([ts[i] for i in anomalies])))
    else:
        print("No Anomaly has been found!")


def test_s_h_esd_remote_abs():
    # Fetch time series
    vals, ts = loop.run_until_complete(
        get_time_series("graphite.del.zillow.local", 80, "sumSeries(zdc.trends.production.pre.*.rum.homedetails-forsale.contactformrender.mobile.turnstile.in)", "-3d")
    )
    anomalies = detect_anoms_s_h_esd(vals, alpha=0.025, frequency=1440)

    if anomalies:
        print("Acutally Found: {}".format(str([ts[i] for i in anomalies])))
    else:
        print("No Anomaly has been found!")


def main():
    print("===== S-H-ESD =====")
    test_s_h_esd_simple()

    print("===== ARIMA =====")
    test_arima_esd_simple()

    print("===== remote raito =====")
    test_s_h_esd_remote_ratio()

    print("===== remote abs =====")
    test_s_h_esd_remote_abs()


if __name__ == '__main__':
    main()