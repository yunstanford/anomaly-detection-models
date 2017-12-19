import numpy as np
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
from numpy import newaxis
import os
import aiohttp
import asyncio
import copy

# reslove import issues in macos
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data(ts_series, seq_len, batch_size=1, ratio=0.9):
    sequence_length = seq_len + 1
    total_len = len(ts_series)
    ts_seqs = []
    for i in range(total_len - sequence_length):
        ts_seqs.append(ts_series[i: i + sequence_length])
    ts_seqs = np.array(ts_seqs)
    N = int(round(ratio * ts_seqs.shape[0]))
    train_set = ts_seqs[:N, :]
    test_set = ts_seqs[N, :]
    # Let's shuffle training set...
    np.random.shuffle(train_set)
    x_train = train_set[:N, :-1]
    y_train = train_set[:N, -1]

    x_test = train_set[N:, :-1]
    y_test = train_set[N:, -1]

    # reshap input to be [samples, time step, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return (x_train, y_train, x_test, y_test)


def build_model(neuron=10, seq_len=1, dropout=0.2, activation="linear", loss="mse", optimizer="rmsprop"):
    model = Sequential()

    # First Layer LSTM
    model.add(
        LSTM(neuron, input_shape=(seq_len, 1), return_sequences=True)
    )
    model.add(Dropout(dropout))

    # # Second Layer LSTM
    model.add(
        LSTM(neuron, input_shape=(seq_len, 1), return_sequences=False)
    )
    model.add(Dropout(dropout))

    # # Flatten
    # # model.add(Flatten())

    # # Feeds to fully connected normal layer
    model.add(Dense(1, activation=activation))

    # Compile model
    model.compile(loss=loss, optimizer=optimizer)

    return model


def normalize(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    return dataset


async def fetch_remote(graphite_host, graphite_port, metric, frm):
    url = "http://{host}:{port}/render?target={metric}&frm={frm}&format=json".format(
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


def predict_single(model, ts_data):
    return model.predict(np.reshape([ts_data], (1, len(ts_data), 1)))[0,0]


# TODO...
def predict_multiple(model, ts_data, predict_steps=1):
    predict_vals = []
    for i in range(predict_steps):
        val = predict_single(model, ts_data)
        predict_vals.append(val)
        ts_data = np.insert(ts_data, len(ts_data), val)
        ts_data = ts_data[1:]
    return predict_vals

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    plt.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
    fig.savefig('lstm.png')


loop = asyncio.get_event_loop()


def main():
    # Params
    epochs  = 1
    seq_len = 50
    batch_size = 250

    # Fetch time series
    vals, ts = loop.run_until_complete(
        get_time_series("graphite.del.zillow.local", 80, "sumSeries(zdc.metrics.production.pre.*.*.rum.school-schoolsearchpage.domready.desktop.turnstile.in)", "-4d")
    )

    original_vals = copy.deepcopy(vals)
    # Normalize Data
    # vals = normalize(vals.reshape(-1, 1))
    print(vals)

    # load data
    X_train, y_train, X_test, y_test = load_data(vals, seq_len=seq_len, batch_size=batch_size)

    # Build and Compile Model
    model = build_model(neuron=100, seq_len=seq_len)

    # Training
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.05
    )

    # predict full
    # predicted = predict_multiple(model, [vals], 10)

    # predict next data point
    # predict a single value
    print(vals[-seq_len:])
    predicted = predict_single(model, vals[-seq_len:])
    print(predicted)

    # predict multiple vals
    predict_vals = predict_multiple(model, vals[-seq_len:], predict_steps=20)
    print(predict_vals)

    # plot results
    # plot_results(predict_vals, original_vals)
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    plt.plot(ts, original_vals, label='True Data')
    # plt.plot(predict_vals, label='Prediction')
    plt.legend()
    plt.show()
    fig.savefig('lstm.png')


if __name__ == '__main__':
    main()
