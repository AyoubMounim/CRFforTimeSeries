# -*- coding: utf-8 -*-

import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from statistics import mean

sns.set_theme()

def seriesToSupervised(data, n_in = 1, n_out = 1, dropNaN = True):
    df = pd.DataFrame(data)
    col, names = [], []
    for i in range(n_in, 0, -1):
        col.append(df.shift(i))
        names.append(f't - {i}')
    for i in range(n_out):
        col.append(df.shift(-i))
        if i == 0:
            names.append('t')
        else:
            names.append(f't + {i}')
    agg = pd.concat(col, axis = 1)
    agg.columns = names
    if dropNaN:
        agg.dropna(inplace = True)
    agg.reset_index(drop = True, inplace = True)
    return agg.values

def trainTestSplit(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]

def forwardValidation(data, n_test, n_out = 1, n_estimators = 1000, delta = 0.1):
    predictions = []
    scores = []
    train, test = trainTestSplit(data, n_test)
    hist = [x for x in train]
    for i in range(len(test)):
        testX, testY = test[i, :-n_out], test[i, -n_out:]
        yhat = randomForestForecast(hist, testX, n_out, n_estimators)
        predictions.append(yhat)
        scores.append(abs(testY-yhat))
        hist.append(test[i])
    alpha = findConformalInterval(scores, delta)
    error = np.sqrt(mean_squared_error(test[:,-n_out:], predictions))
    return [error, alpha]#, test[:, -n_out], predictions

def randomForestForecast(train, testX, n_out = 1, n_estimators = 1000):
    train = np.asarray(train)
    trainX, trainY = train[:, :-n_out], train[:, -n_out:]
    model = RandomForestRegressor(n_estimators)
    if n_out == 1:
        trainY = np.ravel(trainY)
    model.fit(trainX, trainY)
    yhat = model.predict([testX])
    return yhat[0]

def gridSearch(data, n_test, n_estimators_range, step, n_in_range, n_out = 1):
    best_error = 0
    for n_estimator in range(n_estimators_range[0], n_estimators_range[1]+1, step):
        print(f'training forest with {n_estimator} estimators')
        for n_in in range(n_in_range[0], n_in_range[1]+1):
            hist = seriesToSupervised(data = data['Close Price'], n_in = n_in, n_out = n_out)
            hist = np.concatenate((np.reshape(np.asarray(data[f'{ma_periods}-bar Moving Average'])[n_in-1:-1], (len(data.index)-n_in, 1)), hist), axis = 1)
            if n_in < ma_periods:
                hist = hist[ma_periods-n_in:]
            error, alpha = forwardValidation(data = hist, n_test = n_test, n_out = n_out, n_estimators = n_estimator)
            if not best_error:
                best_error = error
                best_in = n_in
                best_est = n_estimator
                best_alpha = alpha
            else:
                if error < best_error:
                    best_error = error
                    best_in = n_in
                    best_est = n_estimator
                    best_alpha = alpha
    best_params = (best_est, best_in, best_error, best_alpha)
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

def movingAverage(data, n_periods, exponential = False):
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
ma_periods = 5

symbol = 'AAPL'
start = '2022-02-01'
end = '2022-02-02'
history = yf.Ticker(symbol).history(start= start, end= end, interval='5m')
history = history[['Close']]
history.rename(columns = {'Close': 'Close Price'}, inplace = True)
history['Time'] = pd.to_datetime(history.index).time
for i in history.index:
    history.loc[i, 'Time'] = str(history.loc[i, 'Time'])
history.set_index('Time', inplace=True)
history[f'{ma_periods}-bar Moving Average'] = movingAverage(data = history['Close Price'], n_periods = ma_periods, exponential=True)
fig, ax = plt.subplots(figsize = (18, 8))
sns.lineplot(data = history, palette = ['darkblue', 'darkorange'])
plt.title(f'{symbol} One Day Time Series for {start} (5 minutes intervals)')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.xticks(rotation=90)
plt.show()

n_test = int(len(history.index)*0.3)

best_parameters = gridSearch(data = history, n_test = n_test, n_estimators_range = (100, 1500), step = 100, n_in_range = (1,10), n_out = n_out)
n_estimators = best_parameters[0]
n_in = best_parameters[1]
alpha = best_parameters[3] 

best_model = RandomForestRegressor(n_estimators)
train = seriesToSupervised(data = history['Close Price'], n_in = n_in, n_out = n_out)
train = np.concatenate((np.reshape(np.asarray(history[f'{ma_periods}-bar Moving Average'])[n_in-1:-1], (len(history.index)-n_in, 1)), train), axis = 1)
if n_in < ma_periods:
    train = train[ma_periods-n_in:]
trainX, trainY = train[:,:-n_out], train[:,-n_out:]
best_model.fit(trainX, np.ravel(trainY))

start= '2022-02-02'
end= '2022-02-03'
history = yf.Ticker(symbol).history(start= start, end= end, interval='5m')
history = history[['Close']]
history.rename(columns = {'Close': 'Close Price'}, inplace = True)
history['Time'] = pd.to_datetime(history.index).time
for i in history.index:
    history.loc[i, 'Time'] = str(history.loc[i, 'Time'])
history.set_index('Time', inplace=True)
history[f'{ma_periods}-bar Moving Average'] = movingAverage(data = history['Close Price'], n_periods = ma_periods, exponential = True)
fig, ax = plt.subplots(figsize = (18, 8))
sns.lineplot(data = history, palette = ['darkblue', 'darkorange'])
plt.title(f'{symbol} One Day Time Series for {start} (5 minutes intervals)')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.xticks(rotation=90)
plt.show()

history['Prediction'] = np.NaN
history['Prediction upper limit'] = np.NaN
history['Prediction lower limit'] = np.NaN
start = max(ma_periods, n_in)
for i in range(start, len(history.index)-1):
    data = seriesToSupervised(history['Close Price'][start-n_in:i+1], n_in = n_in, n_out = n_out)
    ma_element = np.reshape(np.asarray(history[f'{ma_periods}-bar Moving Average'])[max(ma_periods, n_in)-1:i], (len(data), 1))
    data = np.concatenate((ma_element, data), axis = 1)
    trainX, trainY = data[:, :-n_out], data[:, -n_out:]
    best_model.fit(trainX, np.ravel(trainY))
    prediction_vector = [history.iloc[i, 1]]
    for j in range(n_in - 1, -1, -1):
        prediction_vector.append(history.iloc[i-j,0])
    yhat = best_model.predict([prediction_vector])
    history.iloc[i+1, 2] = yhat[0]
    history.iloc[i+1, 3] = yhat[0] + alpha
    history.iloc[i+1, 4] = yhat[0] - alpha
    
fig, ax = plt.subplots(figsize = (18, 8))
ax = sns.lineplot(data = [history['Close Price'], history['Prediction']], palette = ['darkblue', 'firebrick'])
ax = sns.lineplot(data = history['Prediction upper limit'], legend = False, color = 'black', alpha = 0.2)
ax = sns.lineplot(data = history['Prediction lower limit'], legend = False, color = 'black', alpha = 0.2)
l_plus = ax.lines[4]
l_minus = ax.lines[5]
x = l_minus.get_xdata()
y_plus = l_plus.get_ydata()
y_minus = l_minus.get_ydata()
ax.fill_between(x, y_minus, y_plus, color = "turquoise", alpha=0.2)
plt.title(f'{symbol} One Day Time Series (5 minutes intervals)')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.xticks(rotation=90)
plt.show()

error_rate = rateOfError(history['Close Price'], history['Prediction'], alpha = alpha)      
print(error_rate)