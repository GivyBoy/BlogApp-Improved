import pandas as pd
import yfinance as yf
import numpy as np
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""# Portfolio Optimization Web App""")
st.write("""## Created by Anthony Givans""")
st.write(" The Modern Portfolio Theory, put forward by Harry Markowitz, introduced the idea that one should analyze investments in portfolios (groups of assets), instead of singularly. In addition, the theory also showed investors how they could optimize a portfolio. In essence, which portfolio generates the greatest expected return for a given level of risk. Graphically, this is represented by the Efficient Frontier.")
st.write(" ")
st.write(" In this project, I examine whether the MPT is still applicable today and if it can outperform a buy and hold, stock-based, equal-weighted portfolio. I show that the MPT portfolio, based on the Efficient Frontier, consistently outperforms the equal-weighted portfolio and the S&P500, no matter the inputs.")
st.write(" ")
st.write(" **The Sharpe Ratio is used to categorize ‘expected return for a given level of risk’.")
st.write(" ")
st.write(" As an example, you may adjust the slider to the amount that suits you, then copy and paste the following stock tickers in the input box (also, give it some time to load): aapl, tsla, msft, zm, amzn, jpm, gm, nflx, googl, fb")
st.write(" ")

if st.checkbox('Tap/Click to see examples of stock tickers'):
  examples = pd.read_csv('stocks.csv')
  st.write(examples)

investment = st.slider('Investment Amount', 0, 100000, step=5000)
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

stocks = stock_data.columns.values

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
  
  number_of_portfolios = 3000
  RF = pd.read_html("https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/textview.aspx?data=yield")[1]["10 yr"][5]
  portfolio_returns = []
  portfolio_risk = []
  sharpe_ratio_port = []
  portfolio_weights = []
  
  st.write("""## Efficient Markets Frontier""")
  st.write(f' The number of portfolios being used is: {number_of_portfolios}')
  st.write(" Data from 2015 - 2019 is being used in the analysis below.")
  st.write(f" The following graphs give a visual representation of all {number_of_portfolios} portfolios that were computed.")

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
  plt.scatter(portfolio_risk, portfolio_returns, c=portfolio_returns/portfolio_risk,cmap='YlGnBu', marker='.', alpha=0.8)
  plt.xlabel('Volatility (%)')
  plt.ylabel('Expected Returns (%)')
  plt.title('Portfolio Performance')
  plt.colorbar(label='Sharpe ratio')
  st.pyplot()

  #plot data with indicators
  plt.figure(figsize=(10, 5))
  plt.scatter(portfolio_risk, portfolio_returns, c=portfolio_returns/portfolio_risk,cmap='YlGnBu', marker='.', alpha=0.8)
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
  
  DailyReturns = pd.DataFrame(daily_simple_returns_equal)

  updatedW = Highest_sharpe_port['Portfolio Weights']

  weighted = []
  equal = []
  for i in range(len(DailyReturns.index)):
      equal.append(np.sum((DailyReturns.iloc[i] * weights)))
      weighted.append(np.sum((DailyReturns.iloc[i] * updatedW)))
  
  DailyReturns['Portfolio Return - Equal'] = equal
  DailyReturns['Portfolio Return - Weighted'] = weighted
  DailyReturns = DailyReturns/100
  DailyReturns['Portfolio Return - Equal'] = round(DailyReturns['Portfolio Return - Equal']*investment, 2)
  DailyReturns['Portfolio Return - Weighted'] = round(DailyReturns['Portfolio Return - Weighted']*investment, 2)


  e_value = [investment]
  w_value = [investment]

  for i in range(len(DailyReturns.index)):
      e_value.append(round((e_value[i])+DailyReturns['Portfolio Return - Equal'][i],2))
      w_value.append(round((w_value[i])+DailyReturns['Portfolio Return - Weighted'][i],2))
    
  e_value = e_value[1:]
  w_value = w_value[1:]

  DailyReturns['Portfolio Value - Equal'] = e_value
  DailyReturns['Portfolio Value - Weighted'] = w_value
  

  plt.figure(figsize=(10, 5))
  plt.plot(DailyReturns['Portfolio Value - Weighted'], c='red', label='Weighted Portfolio Value')
  plt.plot(DailyReturns['Portfolio Value - Equal'], c='black', label='Equal Weighted Portfolio Value')
  plt.title('BACKTESTING - Portfolio Values Overtime')
  plt.xlabel('Date')
  plt.ylabel('Portfolio Value ($USD)')
  plt.legend()
  st.pyplot()

  start = '2020-01-01'
  end = datetime.today().strftime("%Y-%m-%d")

  s_p = yf.download("^GSPC", start=start, end=end)['Adj Close']
  returns_SP = s_p.pct_change()
  returns_SP = returns_SP*investment
  DailyReturns["S&P500"] = returns_SP

  log_returns = DailyReturns.copy()
  log_returns = log_returns[['Portfolio Value - Equal', 'Portfolio Value - Weighted']]
  
  log_returns['S&P500'] = returns_SP
  log_returns['S&P500'][0] = 0

  sp_value = [investment]

  for i in range(len(log_returns.index)):
      sp_value.append(round((sp_value[i])+log_returns['S&P500'][i],2))
  sp_value = sp_value[1:]

  log_returns['S&P500 Value'] = sp_value

  del log_returns['S&P500']

  plt.figure(figsize=(10, 5))
  plt.plot(log_returns['Portfolio Value - Weighted'], c='red', label='Weighted Portfolio Value')
  plt.plot(log_returns['Portfolio Value - Equal'], c='black', label='Equal Weighted Portfolio Value')
  plt.plot(log_returns['S&P500 Value'], c='blue', label='S&P500')
  plt.title('BACKTESTING - Benchmark Returns')
  plt.xlabel('Date')
  plt.ylabel('Portfolio Value ($USD)')
  plt.legend(loc='lower right')
  st.pyplot()

  if log_returns['Portfolio Value - Weighted'][-1] > log_returns['Portfolio Value - Equal'][-1]:
    st.write(f"The model was right! You would've made ${round(log_returns['Portfolio Value - Weighted'][-1] - log_returns['Portfolio Value - Equal'][-1], 2)} more than the equally weighted portfolio and ${round(log_returns['Portfolio Value - Weighted'][-1] - log_returns['S&P500 Value'][-1], 2)} more than the S&P500! Meaning, you would've outperformed the most uniquitous benchmark in Finance by {round(((log_returns['Portfolio Value - Weighted'][-1] - log_returns['S&P500 Value'][-1])/log_returns['S&P500 Value'][-1]), 2)*100}%!")
  else:
    st.write(' The pandemic really hit hard! Not even the model could have predicted it.')

  st.write('')
  st.write(' Created by Anthony Givans')
else:
  st.write()








