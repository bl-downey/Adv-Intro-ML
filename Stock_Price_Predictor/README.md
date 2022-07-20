# Stock Price Predictor

This project was done for The University of Washington's Professional Master's Program Advanced Introduction to Machine Learning course taught by Dr. Karthik Mohan.

## Introduction

This project, as indicated by the name, was a stock price predictor which informed the user whether it was a good time to buy, sell or hold. This was a partner project and we chose to use 5 stocks from the NASDAQ, namely AMZN, GOOGL, AAPL, NVDA and AMD. There were two result sections for this assignment, backtesting and live trading. 

## Dataset

To acquire data for this project, we created Alpaca API accounts and pulled stock closing price data for the last year into the python notebook we were working with.

For the baseline methods this was all the work that was needed. However, when it came to the Logistic Regression and Deep Learning Models, I created a method to generate dataframes from this closing stock price data. 

The basic dataframe for the ML models encompassed the features: Closing Price, Simple Moving Average (SMA) and Exponential Moving Average (EMA), deviation of closing price from SMA, calculated upper and lower bound of deviation, and slope calculated from the last two days of SMA values which was a smooth curve that prevented halucinations in overall slope.
For more complex dataframe testing, I added functionality to perform feature stacking, which meant that the previous 4 days of closing prices, SMA, EMA and slope values were stacked along the feature dimension of the dataframe. This shrunk the datatable by 4 samples, but allowed a more complex feature space to develop for the models.

Since we needed labels for the data, I wrote an algorithm that detected valleys (opposite of a peak detector) in the stock prices. An example of the labeled data can be seen graphed below for AMZN for a period of 5 months.

![labeled_data](https://user-images.githubusercontent.com/72525765/180095809-5726dd78-b28d-4866-a952-cba00df83b4f.PNG)

## Models

To tackle this very complex problem, we decided to first make a few baseline models. These included models such as Simple Moving Average (SMA), Exponential Moving Average (EMA), and a Crossover Strategy which used both SMA and EMA values. For more complex models we used Seasonal and Trend decomposition using Loess (STL), Logistic Regression (LR), 

### Baseline Models 
Simple Moving Average:

The SMA was calculated as the the mean of the closing stock prices over a given window.

Exponential Moving Average:

The EMA was calculated once and updated with each new stock closing price value. The initial calculation of the EMA used previous days stock closing prices, a variable beta and the mean of the data. The update calculation simply used the existing EMA, beta and the new closing price value. 

Crossover:

The crossover strategy was a simple implementation of predicting buy, sell or hold of a stock. It used the values of SMA for a given day. This method predicted buy if the previous days fast moving average (fSMA) was less than the previous days slow moving average (sSMA) AND the new days fSMA was greater than the new days sSMA. Visualizing that, it predicts buy if the stock is on the rise, and sell if the stock is on the fall, but never is able to predict buy or sell when the stock is truly at the valley or peak of the market.

### More Complex Models

Seasonal and Trend decomposition using Loess:
- *future days:* 10
- *change point prior scale:* 0.05

Logistic Regression (Ridge Classifier from sklearn):

- *Data Normalized:* yes - mean std normalization
- *max_iter:* 800
- *alpha:* [0.01, 0.00009, 0.0009, 0.0009, 0.0009] for stocks ['AMZN', 'GOOGL', 'AAPL', 'NVDA', 'AMD']

Feed Forward Neural Network:

- *API:* Keras Sequenial Model
- *Layer Count:* 5
- *Activation:* ReLU
- *Dropout:* 0.6
- *Loss:* Sparse Categorical Cross Entropy
- *Optimizer:* Adam
- *Target:* 0,1,2 for hold, buy and sell respectively

## Results

## The code
The code notebook is here on the repo and can be used at your preference. I have many imputation and cleaning definitions that could be found useful with similar datasets. Enjoy the code!

## Libraries used
alpaca_trade_api, numpy, pandas, sklearn, tensorflow, and prophet

