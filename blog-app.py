import pandas as pd
import yfinance as yf
import numpy as np
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""# Portfolio Optimization""")
st.write(" This web application seeks to construct the best weighted portfolio, using the Modern Portfolio Theory and the Efficient Markets Hypothesis, given the stock tickers of a portfolio of stocks.")
st.write(" ")
st.write(" As an example, you may adjust the slider to the amount that suits you, then copy and paste the following stock tickers in the input box (also, give it some time to load): aapl, tsla, msft, zm, amzn, jpm, gm, nflx, googl, fb")
st.write(" ")

if st.checkbox('Tap/Click to see examples of stock tickers'):
  examples = pd.read_csv('stocks.csv')
  st.write(examples)

investment = st.slider('Investment Amount', 0, 10000, step=100)
st.write(f'The sum of money being invested is: ${investment} USD')


data= st.text_input('Enter the stocks here: ').upper()
if st.button('Submit'):
  for datum in data:
    if datum == ',':
        data = data.replace(datum, '')
  st.success(f'Your portfolio consists of: {data}; the value is: ${investment} USD ')

stock_data = pd.DataFrame()

start_date = '2015-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')
try:
    stock_data = yf.download(data, start=start_date, end=end_date)['Adj Close']
except:
    st.write('Enter the tickers(for example, AAPL for Apple) for the stocks.')
    break

if len(stock_data.columns) > 0:  
  weights = np.array([1/len(stock_data.columns) for i in range(1, len(stock_data.columns)+1)])
else:
  weights = []

for stock in stock_data.columns.values:
  plt.figure(figsize=(8, 4))
  plt.plot(stock_data[stock])
  plt.title(f"{stock}'s Stock Price ($USD)")
  plt.xlabel('Date')
  plt.ylabel('Stock Price ($USD)')
  plt.legend([stock])
  st.pyplot()

if len(stock_data.columns) > 0: 
  daily_simple_returns = stock_data.pct_change()
  daily_simple_returns = round(daily_simple_returns, 4)*100
  daily_simple_returns_equal = daily_simple_returns[1258:].copy()
  daily_simple_returns_weighted = daily_simple_returns[1258:].copy()

  plt.figure(figsize=(10, 6))
  plt.plot(daily_simple_returns)
  plt.title('Volatility of the Individual Stocks')
  plt.xlabel('Date')
  plt.ylabel('Volatility (%)')
  plt.legend(daily_simple_returns, bbox_to_anchor=(1,1))
  st.pyplot()
  st.write(" The graph above shows the daily fluctuations (as a percent) of the stock price. Simply put, the smaller the spikes, the lower your blood pressure and stress levels will be :)")
  
  number_of_portfolios = 2000
  RF = 0
  portfolio_returns = []
  portfolio_risk = []
  sharpe_ratio_port = []
  portfolio_weights = []
  
  st.write("""## Efficient Markets Frontier""")
  st.write(f' The number of portfolios being used is: {number_of_portfolios}')
  st.write(" Data from 2015 - 2019 is being used in the analysis below.")
  st.write(" The following graphs give a visual representation of all 2000 portfolios that were computed.")

  for portfolio in range(number_of_portfolios):
          # generate a w random weight of length of number of stocks
      weights = np.random.random_sample(len(stock_data.columns))

      weights = weights / np.sum(weights)
      annualized_return = np.sum((daily_simple_returns_equal.mean() * weights) * 252)
      portfolio_returns.append(annualized_return)
          # variance
      matrix_covariance_portfolio = (daily_simple_returns_equal.cov()) * 252
      portfolio_variance = np.dot(weights.T, np.dot(matrix_covariance_portfolio, weights))
      portfolio_standard_deviation = np.sqrt(portfolio_variance)
      portfolio_risk.append(portfolio_standard_deviation)
          # sharpe_ratio
      sharpe_ratio = ((annualized_return - RF) / portfolio_standard_deviation)
      sharpe_ratio_port.append(sharpe_ratio)

      portfolio_weights.append(weights)

  portfolio_risk = np.array(portfolio_risk)
  portfolio_returns = np.array(portfolio_returns)
  sharpe_ratio_port = np.array(sharpe_ratio_port)

  porfolio_metrics = [portfolio_returns, portfolio_risk, sharpe_ratio_port, portfolio_weights]

  portfolio_dfs = pd.DataFrame(porfolio_metrics)
  portfolio_dfs = portfolio_dfs.T
  portfolio_dfs.columns = ['Expected Portolio Returns', 'Portfolio Risk', 'Sharpe Ratio', 'Portfolio Weights']

      # convert from object to float the first three columns.
  for col in ['Expected Portolio Returns', 'Portfolio Risk', 'Sharpe Ratio']:
      portfolio_dfs[col] = portfolio_dfs[col].astype(float)

      # portfolio with the highest Sharpe Ratio
  Highest_sharpe_port = portfolio_dfs.iloc[portfolio_dfs['Sharpe Ratio'].idxmax()]
      # portfolio with the minimum risk
  min_risk = portfolio_dfs.iloc[portfolio_dfs['Portfolio Risk'].idxmin()]

  #plot data without indicators
  plt.figure(figsize=(10, 5))
  plt.scatter(portfolio_risk, portfolio_returns, c=portfolio_returns/portfolio_risk,cmap='YlGnBu', alpha=0.8)
  plt.xlabel('Volatility (%)')
  plt.ylabel('Expected Returns (%)')
  plt.title('Portfolio Performance')
  plt.colorbar(label='Sharpe ratio')
  st.pyplot()

  #plot data with indicators
  plt.figure(figsize=(10, 5))
  plt.scatter(portfolio_risk, portfolio_returns, c=portfolio_returns/portfolio_risk,cmap='YlGnBu', alpha=0.9)
  plt.xlabel('Volatility (%)')
  plt.ylabel('Expected Returns (%)')
  plt.colorbar(label='Sharpe ratio')
  plt.scatter(Highest_sharpe_port['Portfolio Risk'],Highest_sharpe_port['Expected Portolio Returns'], marker=(3,1,0),color='g',s=100, label='Maximum Sharpe ratio' )
  plt.scatter(min_risk['Portfolio Risk'], min_risk['Expected Portolio Returns'], marker=(3,1,0), color='r',s=100, label='Minimum volatility')
  plt.legend(labelspacing=0.8)
  plt.title('Portfolio Performance (with indicators)')
  st.pyplot()


  st.write('-----------------------------------------------------------------------')
  st.write(f"To maximize returns, your Portfolio should consist of: ")
  for i in range(len(Highest_sharpe_port['Portfolio Weights'])):
      st.write(f"{round((Highest_sharpe_port['Portfolio Weights'][i]*100), 2)}% of {stocks[i]}")
  st.write(f"Your Expected Portfolio Returns is: {round(Highest_sharpe_port['Expected Portolio Returns'], 2)}%")
  st.write(f"Your Portfolio Risk is: {round(Highest_sharpe_port['Portfolio Risk'], 2)}%")
  st.write(f"Your Sharpe Ratio is: {round(Highest_sharpe_port['Sharpe Ratio'], 2)}")
  st.write('-----------------------------------------------------------------------')
  st.write(f"To minimize risk, your Portfolio should consist of: ")
  for i in range(len(min_risk['Portfolio Weights'])):
      st.write(f"{round((min_risk['Portfolio Weights'][i]*100), 2)}% of {stocks[i]}")
  st.write(f"Your Expected Portfolio Returns is: {round(min_risk['Expected Portolio Returns'], 2)}%")
  st.write(f"Your Portfolio Risk is: {round(min_risk['Portfolio Risk'], 2)}%")
  st.write(f"Your Sharpe Ratio is: {round(min_risk['Sharpe Ratio'], 2)}")
  st.write('-----------------------------------------------------------------------')

  DailyReturns = pd.DataFrame()
  DailyReturns = daily_simple_returns_equal.sum(axis=1)
  daily_simple_returns_equal['Daily Returns'] = DailyReturns

  port = [122]
  port = pd.DataFrame(port)

  daily_simple_returns_equal['Portfolio Value - Equal'] = port

  daily_simple_returns_equal.iloc[0, -1,] = investment

  for i in range(1, len(daily_simple_returns_equal.index)):
    daily_simple_returns_equal['Portfolio Value - Equal'][i] = daily_simple_returns_equal['Portfolio Value - Equal'][i-1] + daily_simple_returns_equal['Daily Returns'][i-1]

  updatedW = Highest_sharpe_port['Portfolio Weights']
  cols = daily_simple_returns_weighted.columns.values

  AdjustedR = pd.DataFrame()

  for i in range(len(updatedW)):
    Sum = daily_simple_returns_weighted[cols[i]]
    DailyReturnsW = pd.DataFrame(Sum)
    DailyReturns = DailyReturnsW.sum(axis=1) * (1 + round(updatedW[i], 3))
    AdjustedR[f"{cols[i]}'s AR"] = round(DailyReturns, 2)

  TotalAR = AdjustedR.sum(axis=1)

  AdjustedR['Total AR'] = TotalAR

  port2 = [122]
  port2 = pd.DataFrame(port2)

  AdjustedR['Portfolio Value - Weighted'] = port

  AdjustedR.iloc[0, -1,] = investment

  for i in range(1, len(AdjustedR.index)):
    AdjustedR['Portfolio Value - Weighted'][i] = AdjustedR['Portfolio Value - Weighted'][i-1] + AdjustedR['Total AR'][i-1]
  
  st.write("""# BACKTESTING""")
  st.write(" The Weighted Portfolio is what the model above recommends, while the Equal Weighted Portfolio assumes that the portfolio is split evenly among all stocks.")
  st.write(" The data being used is from the start of the trading year in 2020, until the most recent completed trading day. It serves as a means of testing whether the recommendations given by the model is profitable, relative to some other metrics.")
  st.write(" ")  
           
  plt.figure(figsize=(10, 5))
  plt.plot(AdjustedR['Portfolio Value - Weighted'], c='red', label='Weighted Portfolio Value')
  plt.plot(daily_simple_returns_equal['Portfolio Value - Equal'], c='black', label='Equal Weighted Portfolio Value')
  plt.title('BACKTESTING - Portfolio Values Overtime')
  plt.xlabel('Date')
  plt.ylabel('Portfolio Value ($USD)')
  plt.legend()
  st.pyplot()
  st.write(" The graph above shows a comparison of the returns of both portfolios, since the beginning of 2020.")

  logReturns = pd.DataFrame()

  start_date = '2020-01-01'
  end_date = datetime.today().strftime('%Y-%m-%d')
  logReturns['S&P500'] = web.DataReader('^GSPC', data_source='yahoo', start=start_date, end=end_date)['Adj Close']

  port3 = [122]
  port3 = pd.DataFrame(port3)

  logReturns['S&P500 Returns'] = port3
  for i in range(1, len(logReturns.index)):
    logReturns['S&P500 Returns'][i] = logReturns['S&P500'][i] - logReturns['S&P500'][i-1]

  logReturns['S&P500 Portfolio'] = port3
  logReturns.iloc[0, -1,] = investment

  for i in range(1, len(logReturns.index)):
    logReturns['S&P500 Portfolio'][i] = logReturns['S&P500 Portfolio'][i-1] + logReturns['S&P500 Returns'][i]

  logReturns['Adjusted Weights'] = AdjustedR['Portfolio Value - Weighted']
  logReturns['Equal Weights'] = daily_simple_returns_equal['Portfolio Value - Equal']

  logReturns = np.log(logReturns)
           

  plt.figure(figsize=(10, 5))
  plt.plot(logReturns['Adjusted Weights'], c='red', label='Weighted Portfolio Value')
  plt.plot(logReturns['Equal Weights'], c='black', label='Equal Weighted Portfolio Value')
  plt.plot(logReturns['S&P500 Portfolio'], c='blue', label='S&P500')
  plt.title('BACKTESTING - Log Returns')
  plt.xlabel('Date')
  plt.ylabel('Portfolio Value ($USD)')
  plt.legend(loc='lower right')
  st.pyplot()
  st.write(" Both portfolios are being compared to the S&P500, which is the most ubiquitous stock performance benchmark.")
  st.write('')
           
  if AdjustedR['Portfolio Value - Weighted'][-1] > daily_simple_returns_equal['Portfolio Value - Equal'][-1]:
    st.write(f" The model was right! You would've made an extra ${round(AdjustedR['Portfolio Value - Weighted'][-1] - daily_simple_returns_equal['Portfolio Value - Equal'][-1], 2)}!")
  else:
    st.write(' The pandemic really hit hard! Not even the model could have predicted it.')
  
  st.write('')
  st.write(' Created by Anthony Givans')
else:
  st.write()




