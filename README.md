# Conformal Random Forest for Time Series

In this project, a regression model based on a random forest algorithm is used to make conformal predictions of the time series consisting of the prices of financial stocks.

## Table of Contents

* [Introduction](#Introduction)
* [Working Example](#Working-Example)
* [Future Improvements](#Future-Improvements)
* [References](#References)

## Introduction

The aim of this project is to develop a machine learning model that can make predictions about the evolution of financial stocks prices. In particular, the model makes [conformal predictions](https://en.wikipedia.org/wiki/Conformal_prediction), which means that the prediction given by the model is not a single value, but rather it is a continuous interval of values. The with of the conformal interval is controlled by the significance level. For instance, choosing a significance level of 0.1, the output of the model will be such that the actual values fall outside of the conformal interval at most 10% of the times. Once the significance level is defined, the conformal interval is found following the priciple of inductive conforma prediction (ICP). In this project, the nonconformity function is the absolute error, and the nonconformity scores are not normalized, wich means that the conformal interval will be the same for all predictions.

The prediction model is based of a random forest (RF) regressor. To generate a data set that can be used to train a regressor model, we can proceed in the following way. First we fix the number n_in of observations used to make the prediction, and the number n_out of elements to be predicted. Then, given one element of the series, the ensemble consisting of that element, the first (n_in - 1) elements preciding it and the n_out following it, will constitute one row of the data-set. The first n_in colums will be the features, and the rest ones act at the target variables to predict. Repeating this process for all elements in the data series populates our data-set. I also decided to use the [moving average](https://www.investopedia.com/terms/m/movingaverage.asp) and the [RSI](https://www.investopedia.com/terms/r/rsi.asp) time series as extra feature for training.

## Working Example

In this example (see [code](/CRF.py) for the details), I demonstarte how the model performs when it tries to predict the price of Apple stocks in the next 5 minutes. To retrive informations about a particular stock, the [yfinance](https://pypi.org/project/yfinance/) api is used. 

As a first step, I retrieve the time series of the prices for a namber of days, I compute the relevant tecnical indicators (moving average and RSI), and I organize them in a data frame

```python
n_out = 1
ma_period = 5
rsi_period = 5
lag_indicators = {'Moving Average': [ma_period, MovingAverage], 'RSI': [rsi_period, RSI]}
max_lag_period = max([x[0] for x in lag_indicators.values()])
n_features = len(lag_indicators)+2

symbol = 'AAPL'
start_date = '2022-02-07'
end_date = '2022-02-09'
history_data = HistoryDataSet(symbol, start_date, end_date, indicators=lag_indicators)
PlotHistoryDataSet(history_data, symbol, start_date, end_date, columns=(0,2), palette=('blue', 'darkorange'))
```

This is how the time series looks like

![TimeSeries](/Plots/price_history1.png)

This data set, after having appropriately reorganized the time series using the `seriesToSupervised()` function, is then used to train the random forest (RF) regressor. The last 30% of the time series is used as a test set, as well as a calibration set used to compute the confonformal interval. I decided to use 0.1 as significance level. The `GridSearch()` function is used to test different hyperparameters choices, namely the number of estimators of the RF and the number of previous observations used to make a prediction. The output of the function is a tuple containing the bast parameters to use according to a preferred metric, which in my case is the mean squared error, as well as the validation error and the half width of the conformal interval.

```python
test_split = 0.3
delta = 0.1
best_parameters, error, alpha = GridSearch(history_data, test_split, (100, 1000), (1, 10), n_out=n_out, delta=delta)
n_estimators = best_parameters[0]
n_in = best_parameters[1]
print(f'Best parameters: {n_estimators} trees and {n_in} steps, best error: {error}')
best_model = RandomForestRegressor(n_estimators)
```

Now that the model is ready, it can be used to make prediction on a new data set. I retrieve the information of the stocks prices on the following day, and again generate the same type of data frame as before. I use the first n_in prices of the series to predict the stock price in the next 5 minutes, then I retrive the actual value and use it to make the next prediction, and so on until the end of the data set. At each step the model is retrained using the partial time series up to that point

```python
start_date = '2022-02-07'
end_date = '2022-02-10'
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
```

The prediction at each step, along with the upper und lower end of the conformal interval, are stored in the same original time series data frame, so that we can easily confront them

![pred](/Plots/price_pred.png)

The conformal interval is represented by the shaded region in the plot. The root mean squared error for the prediction of this example, when normalized to the mean value of the prices, is 0.2%. Computing the error rate instead, namely the percentage of times in which the actual value is outside of the prediction region, gives an error rate of 4%, which is less that 10%, as expected when remembering that the significance level was set at 0.1. To verify that the model is well behaved with respect to the conformal interval, we can plot the calibration curve of the model. 

```python
history_data = HistoryDataSet(symbol, start_date, end_date, indicators=lag_indicators)
calib = CalibrationCurve(best_parameters, history_data)
fig, ax = plt.subplots(figsize = (12,5))
ax2 = ax.twinx()
ax.set_title('Calibration Curve')
ax.set_ylabel('Average Normalized Interval Width')
ax2.set_ylabel('Average Error Rate')
ax.set_xlabel('Significance Level')
sns.lineplot(x=calib['delta'], y=calib['Average interval with'], color='blue', marker='o', ax=ax)
sns.lineplot(x=calib['delta'], y=calib['Average rate of error'], color='firebrick', marker='o', ax=ax2)
ax.legend(['Interval width'], loc='upper center')
ax2.legend(['Error rate'], loc='upper right')
ax2.yaxis.grid(color='lightgray', linestyle='dashed')
plt.tight_layout()
plt.show()
```

Varing the significance level delta between 0 and 1, the conformal interval is computed on a given calibration set, and the error rate is evaluated of a test set. Given the intrinsic variability of the model, the error rate is computed a number of times for each value of delta, and plotting the average error rate as a function of delta gives the calibration curve

![calibration](/Plots/calib_curve.png)

It is easy to see that the error rate is always lesser or equal to the significance level, i.e., the red curve is always under the 45° degrees line, exatcly as expected. From the plot we can also see how the conformal interval shrinks as the significance level increases, vanishing as delta reaches 1, which is again in agreement with the theoretical expectations.  

## Future Improvements

- Writing more efficient code
- Improving the prediction model by introducing more technical indicators
- Shrinking the conformal interval for more useful predictions

## References

- [Wikipedia](https://en.wikipedia.org/wiki/Conformal_prediction)
- [Investopedia](https://www.investopedia.com)
- [yfinance](https://pypi.org/project/yfinance/)
- [Machine Learning Mastery](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
