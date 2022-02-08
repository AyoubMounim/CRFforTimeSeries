# -*- coding: utf-8 -*-

import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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

def TrainValidateModel(model, data, n_in, n_out=1, test_split=0):
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
    for i in range(testX.shape[0]):
        trainX = np.append(trainX, np.array([testX[i]]), axis=0)
        trainY = np.append(trainY, np.array([testY[i]]), axis=0)
        trainX_scaled = scaler.fit_transform(trainX)
        yhat = model.predict(np.array([trainX_scaled[-1]]))[0]
        pred.append(yhat)
        if n_out == 1:
            model.fit(trainX_scaled, np.ravel(trainY))
        else:
            model.fit(trainX_scaled, trainY)
    pred = np.asarray(pred)
    scores = abs(testY - pred)
    val_error = np.sqrt(mean_squared_error(testY, pred))
    return val_error

def gridSearch(data, test_split, n_estimators_range, n_in_range, n_out = 1, step = 100):
    best_error = 0
    for n_estimator in range(n_estimators_range[0], n_estimators_range[1]+1, step):
        print(f'Training forest with {n_estimator} estimators')
        for n_in in range(n_in_range[0], n_in_range[1]+1):
            model = RandomForestRegressor(n_estimator)
            error = TrainValidateModel(model, data, n_in, n_out=n_out, test_split=test_split)
            print(f'>>> Validation error with {n_in} steps: {error}')
            if not best_error:
                best_error = error
                best_in = n_in
                best_est = n_estimator
            else:
                if error < best_error:
                    best_error = error
                    best_in = n_in
                    best_est = n_estimator
    best_params = (best_est, best_in, best_error)
    return best_params

def findConformalInterval(scores, delta):
    alpha_candidates = []
    for score1 in scores:
        count = 0
        for score2 in scores:
            if score2 < score1:
                count += 1
        test = (count + 1)/(len(scores)+1) - 1 + delta
        if test >= 0:
            alpha_candidates.append(score1)
    return min(np.ravel(alpha_candidates))

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

def rateOfError(data, pred, alpha):
    data = np.asanyarray(data)
    pred = np.asanyarray(pred)
    count = 0
    countNaN = 0
    for i in range(len(data)):
        if np.isnan(pred[i]):
            countNaN +=1
        else:
            if data[i] > pred[i] + alpha or data[i] < pred[i] - alpha:
                count += 1
    return count/(len(data)-countNaN)
    
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
best_parameters = gridSearch(history_data, test_split, (100, 100), (1,5), n_out=n_out)
n_estimators = best_parameters[0]
n_in = best_parameters[1]
error = best_parameters[2]
print(f'Best parameters: {n_estimators} trees and {n_in} steps, best error: {error}')
best_model = RandomForestRegressor(n_estimators)

start_date = '2022-02-01'
end_date = '2022-02-05'
start_pred = 235
history_data = HistoryDataSet(symbol, start_date, end_date, indicators=lag_indicators)
history_for_train, history_for_test = history_data[:start_pred-1], history_data[start_pred-n_in:]
TrainValidateModel(best_model, history_for_train, n_in, n_out=n_out)
history_for_trainX = TrainTestSplit(SeriesToSupervised(history_for_train, n_in, n_out=1))[0]
history_for_trainY = TrainTestSplit(SeriesToSupervised(history_for_train, n_in, n_out=1))[1]
history_for_testX = TrainTestSplit(SeriesToSupervised(history_for_test, n_in, n_out=1))[0]
history_for_testY = TrainTestSplit(SeriesToSupervised(history_for_test, n_in, n_out=1))[1]
scaler = StandardScaler()
history_data['Prediction'] = np.NaN
for i in range(history_for_testX.shape[0]):
    history_for_trainX = np.append(history_for_trainX, np.array([history_for_testX[i]]), axis=0)
    history_for_trainY = np.append(history_for_trainY, np.array([history_for_testY[i]]), axis=0)
    trainX = scaler.fit_transform(history_for_trainX)
    yhat = best_model.predict(np.array([trainX[-1]]))[0]
    history_data.iloc[start_pred+i, n_features] = yhat
    if n_out == 1:
        best_model.fit(trainX, np.ravel(history_for_trainY))
    else:
        best_model.fit(trainX, history_for_trainY)

PlotHistoryDataSet(history_data, columns=[0, n_features], palette=['darkblue', 'firebrick'])
print(history_data[start_pred-1:])
rmse = np.sqrt(mean_squared_error(history_data.iloc[start_pred:, 0], history_data.iloc[start_pred:, n_features]))
mean_price = mean(history_data.iloc[start_pred:,0])
print(f'The normalized root mean squared error of the prediction is: {rmse/mean_price}')
