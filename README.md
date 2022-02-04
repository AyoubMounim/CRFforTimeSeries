# Conformal Random Forest for Time Series
In this project, a regression model based on random forest algorithm is used to make conformal predictions of the evolution of the prices of financial stocks.
To retrive informations about a particular stock, the [yfinance](https://pypi.org/project/yfinance/) api is used. the `yf.Ticker().history()` method retur a data frame with all the information needed about the price time series of a particular stock. Using Apple stocks as an example, this is the 5 minutes time chart of the stock price history

![history_01](/Plots/price_history1.png)

The 5-bars moving average showed in the plot above, is computed from the time series with the function `movingAverage()`, (see [code](/CRF.py)). To generate a data-set that can be used to train a regressor model, we can proceed in the following way. First we fix the number n_in of observations used to make the prediction, then, given one element of the series, the ensemple consisting of that element, first n_in - 1 elements preciding it and the one following it, will constitute one row of the data-set. The first n_in colums will be the features, and the last one acts at the target variable to predict. Repeating this process for all elements in the data series populates our data-set. I also decided to add the moving avearge value corresponding to the element of the data series used to generate the row as an additional feature. The generation of the data-set is managed by the function 
```
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
```
This data set is the used to train the random forest (RF) regressor. The data-set used for train and validation is obdatained from the data series visualized in the plot above. the last 30% of the time series is used as a test set, as well as a calibration set used to compute the confonformal interval. The non-conformicity funtion used in the calculation of the conformal interval is the absolute error, and the confidence level is 10%. The `gridSearch()` function (see [code](/CRF.py)) is used to test different hyperparameters choices, namely the number of estimators of the RF and the number of previous observations to use to make a prediction. The out put of the function is a tuple containing the bast parameters to use according to a preferred metric, which in my case is the mean squared error. The model is then trained using the best hyperparameters on the whole data-set, and the time series of the following day is used as a working example

![pred](/Plots/price_pred.png)

in this plot the model is used to predic the time series step by step, and the conformal interval in represented by the shaded region. Computing the error rate, namely the percentage of times in which the actual value is outside the prediction region, gives an error of 4%, which is less that 10% as expected. 

## Future Impovement

- Writing more efficient code
- Improving the prediction model by introducing more technical indicators
- Shinking the conformal interval
