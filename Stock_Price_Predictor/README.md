# Stock Price Predictor

This project was done for The University of Washington's Professional Master's Program Advanced Introduction to Machine Learning course taught by Dr. Karthik Mohan.

## Introduction

This project, as indicated by the name, was a stock price predictor which informed the user whether it was a good time to buy, sell or hold. This was a partner project and we chose to use 5 stocks from the NASDAQ, namely AMZN, GOOGL, AAPL, NVDA and AMD. There were two main parts to this assignment, backtesting and live trading. 

## Dataset

To acquire data for this project, we created Alpaca API accounts and pulled stock closing price data for the last year into the python notebook we were working with.

For the baseline methods this was all the work that was needed. However, when it came to the Logistic Regression and deep learning models, I created a method to generate dataframes from this closing stock price data. 

The basic dataframe for the ML models encompassed the features: Closing Price, Simple Moving Average (SMA) and Exponential Moving Average (EMA), deviation of closing price from SMA, calculated upper and lower bound of deviation, and slope calculated from the last two days of SMA values which was a smooth curve that prevented halucinations in overall slope.
For more complex dataframe testing, I added functionality to perform feature stacking, which meant that the previous 4 days of closing prices, SMA, EMA and slope values were stacked along the feature column of the dataframe. This shrunk the datatable by 4 samples, but allowed a more complex feature space to develop for the models.

## Models

### Baseline Models 


### More Complex Models


## Results

After data cleaning, imputation, model training and recording the metrics, I produced the below table which summarizes my findings that kNN out performed the baseline RF and the Keras Sequential NN that I built and trained. 

Optimizing my kNN model was crucial to getting these results. Finding the exact value of n and p for the data was very important to getting good results and it was also an exhaustive search space. In the notebook section Hyper-parameter Searching I peform the search by retraining the kNN model and retesting to get f1 on each test model. The values of n and p in the above section of this readme are the optimized values for the split of data that I was using. 

<img width="469" alt="Screen Shot 2022-07-12 at 7 02 49 PM" src="https://user-images.githubusercontent.com/72525765/178635157-54e5a866-fd83-4b27-a640-dea04ea7e0ba.png">

## The code
The code notebook is here on the repo and can be used at your preference. I have many imputation and cleaning definitions that could be found useful with similar datasets. Enjoy the code!

## Libraries used
alpaca_trade_api, numpy, pandas, sklearn, tensorflow, and prophet

