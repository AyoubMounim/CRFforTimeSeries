# Conformal Random Forest for Time Series

In this project, a regression model based on a random forest algorithm is used to make conformal predictions of the time series consisting of the prices of financial stocks.

## Table of Contents

* [Introduction](#Introduction)
* [Working Example](#Working-Example)
* [Future Improvement](#Future-Improvement)
* [References](#References)

## Introduction

The aim of this project is to develop a machine learning model that can make predictions about the evolution of financial stocks prices. In particular, the model makes [conformal predictions](https://en.wikipedia.org/wiki/Conformal_prediction), which means that the prediction given by the model is not a single value, but rather it is a continuous interval of values. The with of the conformal interval is controlled by the significance level. For instance, choosing a significance level of 0.1, the output of the model will be such that the actual values fall outside of the conformal interval at most 10% of the times. Once the significance level is defined, the conformal interval is found following the priciple of inductive conforma prediction (ICP). In this project, the nonconformity function is the absolute error, and the nonconformity scores are not normalized, wich means that the conformal interval will be the same for all predictions.

The prediction model is based of a random forest (RF) regressor. To generate a data-set that can be used to train a regressor model, we can proceed in the following way. First we fix the number n_in of observations used to make the prediction, and the number n_out of elements to be predicted. Then, given one element of the series, the ensemble consisting of that element, the first (n_in - 1) elements preciding it and the n_out following it, will constitute one row of the data-set. The first n_in colums will be the features, and the rest ones act at the target variables to predict. Repeating this process for all elements in the data series populates our data-set. I also decided to add the [moving avearge](https://www.investopedia.com/terms/m/movingaverage.asp) and the [RSI](https://www.investopedia.com/terms/r/rsi.asp) values corresponding to the element of the data series used to generate the row as additional features.

## Working Example

For this example (see [code](/CRF.py) for the details), I demonstarte how to model performs when predicting what will be the price of Apple stocks in the next 5 minutes. To retrive informations about a particular stock, the [yfinance](https://pypi.org/project/yfinance/) api is used. 

As a first step, I retrieve the time series of the prices for a hole day, I compute the relevant tecnical indicators (moving average and RSI), and I organize them in a data frame

```
symbol = 'AAPL'
start_date = '2022-02-01'
end_date = '2022-02-02'
history = yf.Ticker(symbol).history(start = start_date, end = end_date, interval='5m')
history = history[['Close']]
history.rename(columns = {'Close': 'Close Price'}, inplace = True)
history['Time'] = pd.to_datetime(history.index).time
for i in history.index:
    history.loc[i, 'Time'] = str(history.loc[i, 'Time'])
history.set_index('Time', inplace=True)
history[f'{ma_periods}-bar Moving Average'] = movingAverage(data = history['Close Price'], n_periods = ma_periods, exponential=True)
history[f'{rsi_period}-bar RSI'] = RSI(data = history['Close Price'], n_periods = rsi_period)
```

This is how the time series looks like

![TimeSeries](/Plots/price_history1.png)

This data set, after having appropriately reorganized the time series using the `seriesToSupervised()` function, is then used to train the random forest (RF) regressor. The last 30% of the time series is used as a test set, as well as a calibration set used to compute the confonformal interval. I decided to use 0.1 as significance level. The `gridSearch()` function is used to test different hyperparameters choices, namely the number of estimators of the RF and the number of previous observations used to make a prediction. The output of the function is a tuple containing the bast parameters to use according to a preferred metric, which in my case is the mean squared error, as well as the validation error and the half with of the conformal interval. The model is then retrained using the best hyperparameters on the whole data set.

```
n_test = int(len(history.index)*0.3)
best_parameters = gridSearch(data = history, n_test = n_test, n_estimators_range = (100, 1500), step = 100, n_in_range = (1,10), n_out = n_out)
n_estimators = best_parameters[0]
n_in = best_parameters[1]
alpha = best_parameters[3] 
best_model = RandomForestRegressor(n_estimators)
train = seriesToSupervised(data = history['Close Price'], n_in = n_in, n_out = n_out)
ma_array = np.reshape(np.asarray(history[f'{ma_periods}-bar Moving Average'])[n_in-1:-1], (len(history.index)-n_in, 1))
rsi_array = np.reshape(np.asarray(history[f'{rsi_period}-bar RSI'])[n_in-1:-1], (len(history.index)-n_in, 1))
train = np.concatenate((rsi_array, ma_array, train), axis = 1)
if n_in < ma_periods:
    train = train[ma_periods-n_in:]
trainX, trainY = train[:,:-n_out], train[:,-n_out:]
best_model.fit(trainX, np.ravel(trainY))
```

Now that the model is ready, it can be used to make prediction on a new data set. I retrieve the information of the stocks prices on the following day, and again generate the same type of data frame as before. I use the first n_in prices of the series to predict the stock price in the next 5 minutes, then I retrive the actual value and use it to make the next prediction, and so on until the end of the data set. At each step the model is retrained using the partial time series up to that point

```
start_date = '2022-02-02'
end_date = '2022-02-03'
history = yf.Ticker(symbol).history(start = start_date, end = end_date, interval='5m')
history = history[['Close']]
history.rename(columns = {'Close': 'Close Price'}, inplace = True)
history['Time'] = pd.to_datetime(history.index).time
for i in history.index:
    history.loc[i, 'Time'] = str(history.loc[i, 'Time'])
history.set_index('Time', inplace=True)
history[f'{ma_periods}-bar Moving Average'] = movingAverage(data = history['Close Price'], n_periods = ma_periods, exponential = True)
history[f'{rsi_period}-bar RSI'] = RSI(data = history['Close Price'], n_periods = rsi_period)
history['Prediction'] = np.NaN
history['Prediction upper limit'] = np.NaN
history['Prediction lower limit'] = np.NaN
start = max(ma_periods, n_in)
for i in range(start, len(history.index)-1):
    data = seriesToSupervised(history['Close Price'][start-n_in:i+1], n_in = n_in, n_out = n_out)
    ma_array = np.reshape(np.asarray(history[f'{ma_periods}-bar Moving Average'])[max(ma_periods, n_in)-1:i], (len(data), 1))
    rsi_array = np.reshape(np.asarray(history[f'{rsi_period}-bar RSI'])[max(ma_periods, n_in)-1:i], (len(data), 1))
    data = np.concatenate((rsi_array, ma_array, data), axis = 1)
    trainX, trainY = data[:, :-n_out], data[:, -n_out:]
    best_model.fit(trainX, np.ravel(trainY))
    prediction_vector = [history.iloc[i, 2], history.iloc[i, 1]]
    for j in range(n_in - 1, -1, -1):
        prediction_vector.append(history.iloc[i-j,0])
    yhat = best_model.predict([prediction_vector])
    history.iloc[i+1, 3] = yhat[0]
    history.iloc[i+1, 4] = yhat[0] + alpha
    history.iloc[i+1, 5] = yhat[0] - alpha
```

The prediction at each step, along with the upper und lower end of the conformal interval, are stored in the same original time series data frame, so that we can easily confront them

![pred](/Plots/price_pred.png)

The conformal interval is represented by the shaded region in the plot. Computing the error rate, namely the percentage of times in which the actual value is outside the prediction region, gives an error of 4%, which is less that 10% as expected. 

## Future Improvement

- Writing more efficient code
- Improving the prediction model by introducing more technical indicators
- Shrinking the conformal interval for more useful predictions

## References

- [Wikipedia](https://en.wikipedia.org/wiki/Conformal_prediction)
- [Investopedia](https://www.investopedia.com)
- [yfinance](https://pypi.org/project/yfinance/)
- [Machine Learning Mastery](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
