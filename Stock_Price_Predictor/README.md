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

The SMA was calculated as the the mean of the closing stock prices over a given window. This algorithm predicted buy when the mean began to increase, and sell when the mean began to decrease. This method was not very effective as it always lagged behind the actual stock price.

Exponential Moving Average:

The EMA was calculated once and updated with each new stock closing price value. The initial calculation of the EMA used previous days stock closing prices, a variable beta and the mean of the data. The update calculation simply used the existing EMA, beta and the new closing price value. The buying and selling of the stock saw little improvement with this method versus the previous, this is because the exponential nature of the method gave better trend lines for a buy and sell predictor.

Crossover:

The crossover strategy was a simple implementation of predicting buy, sell or hold of a stock. It used two averages, a fast moving and a slow moving average, which either of these averages could be SMA or EMA. This method predicted buy if the previous days fast moving average (fSMA or fEMA) was less than the previous days slow moving average (sSMA or sEMA) AND the new days fSMA/fEMA was greater than the new days sSMA/sEMA. Visualizing that, it predicts buy if the stock is on the rise, and sell if the stock is on the fall, but never is able to predict buy or sell when the stock is truly at the valley or peak of the market.

### More Complex Models

Seasonal and Trend decomposition using Loess:
- *future days:* 10
- *change point prior scale:* 0.05

Logistic Regression (Ridge Classifier from sklearn):

- *Data Normalized:* yes - mean std normalization
- *max_iter:* 800
- *alpha:* [0.01, 0.00009, 0.0009, 0.0009, 0.0009] for stocks ['AMZN', 'GOOGL', 'AAPL', 'NVDA', 'AMD']

Feed Forward Neural Network (Dense):

- *API:* Keras Sequenial Model
- *Layer Count:* 5
- *Activation:* ReLU
- *Dropout:* 0.6
- *Loss:* Sparse Categorical Cross Entropy
- *Optimizer:* Adam
- *Target:* 0,1,2 for hold, buy and sell respectively

## Results

Example snippets of buy/sell graphs for the strategies can be found below, all showing the same example stock: AMZN. 

Crossover fSMA/sSMA: As you can see below, this crossover strategy was not performing as well as had hoped, but it did manage to buy the stock and sell it at a higher price before the market when down. Although this method appears to buy and sell at non ideal times, it accumulated a net 1.3% profit. 

![crossover_amzn](https://user-images.githubusercontent.com/72525765/180101545-c6d9436f-87d0-4ea0-bb10-a776e1c11d6f.PNG)

Crossover fEMA/sSMA: Similarly to the above graph, this crossover strategy was also not very effective. However, it does make certain buys and sells at good times and thus did result in a positive net profit, but only ever so slightly.

![crossover_amzn_emasma](https://user-images.githubusercontent.com/72525765/180102347-d419a7c4-1624-48eb-a7a5-d2c938d6b81f.PNG)

Crossover fEMA/sEMA: This strategy performed about on par with the first crossover strategy but made a signifcantly greater amound of buys and sells. There are some buy and sell decisions in this graph that are very good points to buy and sell, but conversely this algorithm also made a lot of poor predictions, causing the overall profits to only be 1.25%.

![crossover_amzn_emaema](https://user-images.githubusercontent.com/72525765/180102616-915ccaee-7ebc-4f29-96f4-99f870d70fd9.PNG)

STL: STL showed much greater promise with the buy and sell pattern. With a total of 6.42% profit gain from this buy/sell pattern, STL did a much better job of buying low and selling high. This is heavily based on the type of algorithm used, as STL is a much more complicated algorithm from Facebooks Prophet API. 

![stl_amzn](https://user-images.githubusercontent.com/72525765/180103241-ae3150d2-bc65-4e8d-9990-5720a1c40db0.PNG)

Logistic Regression: The dataframe that I created for this model was using feature stacking. As can be seen below, the LR model was a heavy 'buy on the downslope' type of algorithm. To counter this effect, I implemented an algorithm to perform dollar cost averaging, so it buys heavier as it goes down the slope. This model doesn't always buy at the best times, nor sell at the best times, but it did manage to achieve a 7.06% profit on the stock. 

![LR_amzn](https://user-images.githubusercontent.com/72525765/180103565-59a446e1-fc4e-46ef-9bf8-2c2ae8d85192.PNG)

Dense Net: The dense network for this stock was not as successful as other stocks, however it was still a positive percent profit at 1.56%. Since the model also has a similar buying pattern to the LR model, I implemented dollar cost averaging to this model as well. 

![densenet_amzn](https://user-images.githubusercontent.com/72525765/180104280-d08f2961-382d-4d01-a2c8-3ab571baaa29.PNG)

Overall, the below profits table shows the most successful algorithm as the LR model at a net gain across all stocks as 3.39% on backtesting. 

![percent_profits](https://user-images.githubusercontent.com/72525765/180104482-2ca26b6f-1e85-4b02-aeea-100a00fd8760.PNG)

## The code
The code notebook (Mini_Project_1_Q1_Backtesting.ipynb) is here on the repo and can be used at your preference. Enjoy the code, and thanks for reading through this!

## Libraries used
alpaca_trade_api, numpy, pandas, sklearn, tensorflow, and prophet

