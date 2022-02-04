# Conformal Random Forest for Time Series
In this project, a regression model based on random forest algorithm is used to make conformal predictions of the evolution of the prices of financial stocks.
To retrive informations about a particular stock, the [yfinance](https://pypi.org/project/yfinance/) api is used. the `yf.Ticker().history()` method retur a data frame with all the information needed about the price time series of a particular stock. Using Apple stocks as an example, this is the 5 minutes time chart of the stock price history

