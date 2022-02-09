# -*- coding: utf-8 -*-

import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from statistics import mean

sns.set_theme()

def HistoryDataSet(symbol, start_date, end_date, interval='5m', indicators = {}):
    history = yf.Ticker(symbol).history(start=start_date, end=end_date, interval=interval)
    history = history[['Close', 'Volume']]
    history.rename(columns={'Close': 'Close Price'}, inplace=True)
    '''
    history['Time'] = pd.to_datetime(history.index).time
    for i in history.index:
        history.loc[i, 'Time'] = str(history.loc[i, 'Time'])
    history.set_index('Time', inplace=True)
    '''
    history.reset_index(drop=True, inplace=True)
    if indicators:
        for key in indicators.keys():
            history[f'{indicators[key][0]}-bar {key}'] = indicators[key][1](data=history['Close Price'], n_periods=indicators[key][0])
    return history

def PlotHistoryDataSet(data, columns = [], palette = []):
    fig, ax = plt.subplots(figsize=(18, 8))
    if columns:
        data_list = []
        for i in columns:
            data_list.append(data.iloc[:, i])
        if palette:
            if len(columns)==len(palette):
                sns.lineplot(data=data_list, palette=palette)
                plt.title(f'One Day Time Series (5 minutes intervals)')
                plt.xlabel('Time')
                plt.xticks(rotation=90)
                plt.show()
                return
            else:
                raise ValueError('lenght of palette must equal lenght of columns to show')
        else:
            sns.lineplot(data=data_list)
            plt.title(f'One Day Time Series (5 minutes intervals)')
            plt.xlabel('Time')
            plt.xticks(rotation=90)
            plt.show()
            return
    if palette:
        if len(data.columns)==len(palette):
            sns.lineplot(data=data, palette=palette)
            plt.title(f'One Day Time Series (5 minutes intervals)')
            plt.xlabel('Time')
            plt.xticks(rotation=90)
            plt.show()
            return
        else:
            raise ValueError('lenght of palette must equal lenght of columns to show')
    sns.lineplot(data=data)
    plt.title(f'One Day Time Series (5 minutes intervals)')
    plt.xlabel('Time')
    plt.xticks(rotation=90)
    plt.show()
    return

def SeriesToSupervised(data, n_in = 1, n_out = 1, dropNaN = True):
    df = pd.DataFrame(data)
    col = []
    for i in range(n_in, 0, -1):
        col.append(df.shift(i))
    for i in range(n_out):
        col.append(df.iloc[:,0].shift(-i))
    agg = pd.concat(col, axis = 1)
    if dropNaN:
        agg.dropna(inplace=True)
    agg.reset_index(drop=True, inplace=True)
    return agg.values

def TrainTestSplit(data, test_split = 0, n_out=1):
    if not test_split:
        train = data[:, :]
        trainX, trainY = train[:, :-n_out], train[:, -n_out:]
        return trainX, trainY
    n_test = int(len(data) * test_split)
    train, test = data[:-n_test, :], data[-n_test:, :]
    trainX, trainY = train[:, :-n_out], train[:, -n_out:]
    testX, testY = test[:, :-n_out], test[:, -n_out:]
    return trainX, trainY, testX, testY

def TrainValidateModel(model, data, n_in, test_split=0, n_out=1, delta=0):
    supervised_data = SeriesToSupervised(data, n_in, n_out=n_out)
    if not test_split:
        trainX, trainY = TrainTestSplit(supervised_data, test_split=test_split, n_out=n_out)
        scaler = StandardScaler()
        trainX = scaler.fit_transform(trainX)
        if n_out == 1:
            trainY = np.ravel(trainY)
        model.fit(trainX, trainY)
        return
    pred = []
    trainX, trainY, testX, testY = TrainTestSplit(supervised_data, test_split=test_split, n_out=n_out)
    scaler = StandardScaler()
    trainX_scaled = scaler.fit_transform(trainX)
    if n_out==1:
        model.fit(trainX_scaled, np.ravel(trainY))
    else:
        model.fit(trainX_scaled, trainY)
    scores = []
    mse_cumulative = 0
    for i in range(testX.shape[0]):
        trainX = np.append(trainX, np.array([testX[i]]), axis=0)
        trainY = np.append(trainY, np.array([testY[i]]), axis=0)
        trainX_scaled = scaler.fit_transform(trainX)
        yhat = model.predict(np.array([trainX_scaled[-1]]))
        mse_cumulative += ((trainY[-1]-yhat)**2).sum()
        scores.append((abs(trainY[-1]-yhat).sum())/n_out)
        if n_out == 1:
            model.fit(trainX_scaled, np.ravel(trainY))
        else:
            model.fit(trainX_scaled, trainY)
    val_error = np.sqrt(mse_cumulative/(n_out*testX.shape[0]))
    if delta:
        alpha = FindConformalInterval(scores, delta)
        return val_error, alpha
    return val_error

def gridSearch(data, test_split, n_estimators_range, n_in_range, delta, n_out = 1, step = 100):
    best_error = 0
    for n_estimator in range(n_estimators_range[0], n_estimators_range[1]+1, step):
        print(f'Training forest with {n_estimator} estimators')
        for n_in in range(n_in_range[0], n_in_range[1]+1):
            model = RandomForestRegressor(n_estimator)
            error, alpha = TrainValidateModel(model, data, n_in, n_out=n_out, test_split=test_split, delta=delta)
            print(f'>>> Validation error with {n_in} steps: {error}')
            if not best_error:
                best_error = error
                best_alpha = alpha
                best_in = n_in
                best_est = n_estimator
            else:
                if error < best_error:
                    best_error = error
                    best_alpha = alpha
                    best_in = n_in
                    best_est = n_estimator
    best_params = (best_est, best_in)
    return best_params, best_error, best_alpha

def FindConformalInterval(scores, delta):
    scores.sort()
    scores = np.asarray(scores)
    scores = scores.reshape((scores.shape[0], 1))
    scaler = MinMaxScaler()
    scores = scaler.fit_transform(scores)
    for alpha_s in np.arange(0, 1.1, 0.1):
        count = 0
        for alpha in scores:
            if alpha<alpha_s:
                count += 1
            else:
                break
        test = (count+1)/(scores.shape[0]+1)-1+delta
        if test>=0:
            return np.ravel(scaler.inverse_transform([[alpha_s]]))[0]
    raise ValueError('Cannot find conformal interval')

def MovingAverage(data, n_periods, exponential = False):
    data = np.asarray(data)
    ma = [np.NaN]*(n_periods - 1)
    if exponential:
        ma_value = mean(data[:n_periods])
        ma.append(ma_value)
        factor = 2/(n_periods + 1)
        for i in range(n_periods, len(data)):
            ma_value = data[i]*factor + ma[-1]*(1-factor)
            ma.append(ma_value)
        return ma
    for i in range(n_periods, len(data)+1):
        ma_value = mean(data[i-n_periods:i])
        ma.append(ma_value)
    return ma

def RSI(data, n_periods):
    data = np.asarray(data)
    rsi_list = [np.NaN]*(n_periods - 1)
    for i in range(n_periods-1, len(data)):
        gain, loss = [], []
        for j in range(n_periods-1):
            diff = data[i-j] - data[i-j-1]
            if diff >= 0:
                gain.append(diff)
            else:
                loss.append(diff)
        try:
            mean_loss = abs(mean(loss))
        except:
            mean_loss = 0
        try:
            mean_gain = mean(gain)
        except:
            mean_gain = 0
        if mean_loss == 0:
            rsi_list.append(100)
        else:
            rs = mean_gain/mean_loss
            rsi_list.append(100*rs/(1+rs))
    return rsi_list

def RateOfError(data, pred, alpha):
    data = np.asarray(data)
    pred = np.asarray(pred)
    if len(data.shape)!=1 or len(pred.shape)!=1 or len(data.shape)!=len(pred.shape):
        raise ValueError('Array must be one dimensional')
    error_count = ((data<(pred-alpha))+(data>(pred+alpha))).sum()
    return error_count*100/data.shape[0]

def CalibrationCurve(model_parameters, data, calibration_split=0.3):
    columns = ['delta', 'Average interval with', 'Average rate of error']
    df = pd.DataFrame(columns=columns)
    n_trees = model_parameters[0]
    n_in = model_parameters[1]
    model = RandomForestRegressor(n_trees)
    n_calibration = int(calibration_split*len(data.index))
    data_calibration = data[:-n_calibration]
    data_test = data[-n_calibration-n_in:]
    for delta in np.arange(0.05, 1.1, 0.05):
        print(f'Processing delta: {delta}')
        ensemble_error = []
        ensemble_alpha = []
        for i in range(11):
            print(f'>>> experiment number: {i}')
            error, alpha = TrainValidateModel(model, data_calibration, n_in, test_split=test_split, delta=delta)
            data_calibrationX, data_calibrationY = TrainTestSplit(SeriesToSupervised(data_calibration, n_in))
            data_calibrationY = np.ravel(data_calibrationY)
            data_testX, data_testY = TrainTestSplit(SeriesToSupervised(data_test, n_in))
            data_testY = np.ravel(data_testY)
            scaler = StandardScaler()
            pred = []
            for i in range(data_testX.shape[0]):
                data_calibrationX = np.append(data_calibrationX, np.array([data_testX[i]]), axis=0)
                data_calibrationY = np.append(data_calibrationY, np.array([data_testY[i]]), axis=0)
                data_calibrationX_scaled = scaler.fit_transform(data_calibrationX)
                yhat = model.predict(np.array([data_calibrationX_scaled[-1]]))[0]
                pred.append(yhat)
                model.fit(data_calibrationX_scaled, data_calibrationY)
            error_rate = RateOfError(np.ravel(data_testY), pred, alpha)
            ensemble_error.append(error_rate)
            ensemble_alpha.append(alpha)
        df = df.append(pd.Series([delta, mean(ensemble_alpha), mean(ensemble_error)], index=columns), ignore_index=True)
    return df

    
n_out = 1
ma_period = 5
rsi_period = 5
lag_indicators = {'Moving Average': [ma_period, MovingAverage], 'RSI': [rsi_period, RSI]}
max_lag_period = max([x[0] for x in lag_indicators.values()])
n_features = len(lag_indicators)+2

symbol = 'AAPL'
start_date = '2022-02-01'
end_date = '2022-02-04'
history_data = HistoryDataSet(symbol, start_date, end_date, indicators=lag_indicators)
PlotHistoryDataSet(history_data, columns=[0,2])

test_split = 0.3
delta = 0.1
best_parameters, error, alpha = gridSearch(history_data, test_split, (100, 100), (1,1), n_out=n_out, delta=delta)
n_estimators = best_parameters[0]
n_in = best_parameters[1]
print('')
print(f'Best parameters: {n_estimators} trees and {n_in} steps, best error: {error}')
best_model = RandomForestRegressor(n_estimators)

#This portion trains the best model on 2 days and predicts the 3th one
start_date = '2022-02-02'
end_date = '2022-02-05'
start_pred = 78*2+1 #there are 78 observations in a one day time series
history_data = HistoryDataSet(symbol, start_date, end_date, indicators=lag_indicators)
history_for_train, history_for_test = history_data[:start_pred-1], history_data[start_pred-n_in:]
TrainValidateModel(best_model, history_for_train, n_in, n_out=n_out)
history_for_trainX = TrainTestSplit(SeriesToSupervised(history_for_train, n_in, n_out=1))[0]
history_for_trainY = TrainTestSplit(SeriesToSupervised(history_for_train, n_in, n_out=1))[1]
history_for_testX = TrainTestSplit(SeriesToSupervised(history_for_test, n_in, n_out=1))[0]
history_for_testY = TrainTestSplit(SeriesToSupervised(history_for_test, n_in, n_out=1))[1]
scaler = StandardScaler()
history_data['Prediction lower limit'] = np.NaN
history_data['Prediction'] = np.NaN
history_data['Prediction upper limit'] = np.NaN
for i in range(history_for_testX.shape[0]):
    history_for_trainX = np.append(history_for_trainX, np.array([history_for_testX[i]]), axis=0)
    history_for_trainY = np.append(history_for_trainY, np.array([history_for_testY[i]]), axis=0)
    trainX = scaler.fit_transform(history_for_trainX)
    yhat = best_model.predict(np.array([trainX[-1]]))[0]
    for j in range(-1,2):
        history_data.iloc[start_pred+i, n_features+1+j] = yhat+j*alpha
    if n_out == 1:
        best_model.fit(trainX, np.ravel(history_for_trainY))
    else:
        best_model.fit(trainX, history_for_trainY)

PlotHistoryDataSet(history_data, columns=[0, n_features, n_features+1, n_features+2], palette=['darkblue', 'black', 'firebrick', 'black'])
rmse = np.sqrt(mean_squared_error(history_data.iloc[start_pred:, 0], history_data.iloc[start_pred:, n_features+1]))
mean_price = mean(history_data.iloc[start_pred:,0])
error_rate = RateOfError(history_data.iloc[start_pred:, 0], history_data.iloc[start_pred:, n_features+1], alpha)
print('')
print(f'The normalized root mean squared error of the prediction is: {rmse/mean_price}\n'
      f'The error rate is: {error_rate}')
exit()
#This portion computes the calibration curve
calib = CalibrationCurve(best_parameters, history_data)