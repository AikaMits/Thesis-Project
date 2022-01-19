import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error

prediction_results = {}


def parse_file_to_dataframe(uploaded_file):
    if uploaded_file == '':
        return None

    df = pd.read_csv(uploaded_file)

    date_column = df.keys()[0]
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)

    return df


def get_prediction_result(prediction_id):

    if prediction_id in prediction_results:
        return prediction_results[prediction_id]

    return {}


def execute_naive(dataset_file, train_year, test_year, prediction_id):
    prediction_results[prediction_id] = {'status': 'processing'}
    df = parse_file_to_dataframe(dataset_file)

    data_column = df.keys()[0]
    train = df.loc[train_year][data_column]
    test = df.loc[test_year][data_column]

    forecast = test.copy()
    forecast[0:] = train[-1]

    response = {'filename': dataset_file}
    response.update(get_metrics(test, forecast))

    response.update({'train_labels': train.index.tolist()})
    response.update({'train': train.values.tolist()})

    response.update({'test_labels': test.index.tolist()})
    response.update({'test': test.values.tolist()})

    response.update({'forecast_labels': forecast.index.tolist()})
    response.update({'forecast': forecast.values.tolist()})
    prediction_results[prediction_id] = response


def execute_snaive(dataset_file, train_year, test_year, prediction_id):
    prediction_results[prediction_id] = {'status': 'processing'}
    df = parse_file_to_dataframe(dataset_file)

    data_column = df.keys()[0]
    train = df.loc[train_year][data_column]
    test = df.loc[test_year][data_column]

    forecast = test.copy()
    forecast[0:] = [(train[i] if train.size > i else train[train.size - 1]) for i in range(forecast.size)]

    response = {'filename': dataset_file}
    response.update(get_metrics(test, forecast))

    response.update({'train_labels': train.index.tolist()})
    response.update({'train': train.values.tolist()})

    response.update({'test_labels': test.index.tolist()})
    response.update({'test': test.values.tolist()})

    response.update({'forecast_labels': forecast.index.tolist()})
    response.update({'forecast': forecast.values.tolist()})
    prediction_results[prediction_id] = response


def execute_arima(dataset_file, train_year, test_year, prediction_id):
    prediction_results[prediction_id] = {'status': 'processing'}
    from statsmodels.tsa.arima.model import ARIMA
    df = parse_file_to_dataframe(dataset_file)

    data_column = df.keys()[0]
    train = df.loc[train_year][data_column]
    test = df.loc[test_year][data_column]

    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)

    forecast = test.copy()
    forecast[0:] = [x for x in predictions]

    response = {'filename': dataset_file}
    response.update(get_metrics(test, forecast))

    response.update({'train_labels': train.index.tolist()})
    response.update({'train': train.values.tolist()})

    response.update({'test_labels': test.index.tolist()})
    response.update({'test': test.values.tolist()})

    response.update({'forecast_labels': forecast.index.tolist()})
    response.update({'forecast': forecast.values.tolist()})
    prediction_results[prediction_id] = response


def execute_lstm(dataset_file, prediction_id):
    prediction_results[prediction_id] = {'status': 'processing'}

    dataframe = pd.read_csv(dataset_file, usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print(len(train), len(test))

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    trainPredictPlot = trainPredictPlot.flatten()
    trainPredictPlot = trainPredictPlot[~np.isnan(trainPredictPlot)]

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    testPredictPlot = testPredictPlot.flatten()
    testPredictPlot = testPredictPlot[~np.isnan(testPredictPlot)]

    forecast = scaler.inverse_transform(dataset).flatten()

    response = {'filename': dataset_file}
    response.update(get_metrics(np.nan_to_num(testPredictPlot), np.nan_to_num(forecast)))

    response.update({'train_labels': [*range(0, dataset.size, 1)]})
    response.update({'train': np.nan_to_num(trainPredictPlot).tolist()})

    response.update({'test_labels': [*range(trainPredictPlot.size, dataset.size, 1)]})
    response.update({'test': np.nan_to_num(testPredictPlot).tolist()})

    response.update({'forecast_labels': [*range(0, dataset.size, 1)]})
    response.update({'forecast': np.nan_to_num(forecast).tolist()})

    prediction_results[prediction_id] = response


def get_metrics(actual_data, forecast_data):
    forecast_error = [actual_data[i] - forecast_data[i] for i in range(0, actual_data.size)]
    mean_absolute_deviation = sum([abs(fe) for fe in forecast_error]) / actual_data.size
    print(f'MPA: ', mean_absolute_deviation)
    mean_square_error = sum([(fe ** 2) for fe in forecast_error]) / actual_data.size
    print(f'MSE: ', mean_square_error)
    mean_absolute_percent_error = (sum([forecast_error[i] / actual_data[i] for
                                    i in range(0, len(forecast_error))]) / actual_data.size) * 100
    print(f'MAPE: ', mean_absolute_percent_error)

    return {'mpa': mean_absolute_deviation,
            'mse': mean_square_error,
            'mape': mean_absolute_percent_error}
