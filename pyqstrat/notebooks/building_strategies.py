#cell 2
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed") # ignore pesky warning, see https://github.com/numpy/numpy/pull/432

import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import os

import pyqstrat as pq

pq.set_defaults() # Set some display defaults to make dataframes and plots easier to look at

try:
    file_path = os.path.dirname(os.path.realpath(__file__)) + '/../examples/bitcoin_1min.csv' # If we are running from unit tests
except:
    file_path = '../examples/bitcoin_1min.csv'
    
prices = pd.read_csv(file_path)

dates = pd.to_datetime(prices.date).values
o = prices.o.values
h = prices.h.values
l = prices.l.values
c = prices.c.values
v = prices['v.usd'].values

#cell 4
md = pq.MarketData(dates, c, o, h, l, v)
md.resample(sampling_frequency='5 min', inplace = True)
md.plot(date_range = ('2018-03-24 09:30', '2018-03-24 16:00'))
md.describe(time_distribution_frequency = '30 mins')

#cell 6
def sma(marketdata, lookback_period): # simple moving average
    return pd.Series(marketdata.c).rolling(window = lookback_period).mean().values

def band(marketdata, lookback_period, num_std, upper):
    std = pd.Series(marketdata.c).rolling(window = 20).std()
    return sma(marketdata, lookback_period = lookback_period) + num_std * std * (1 if upper else -1)

num_std = 1 # Each band is 2 standard deviations away from the moving average of price

lookback_period = 20 # Length of moving average window
mid_band = sma(md, lookback_period)
upper_band = band(md, lookback_period, num_std, upper = True)
lower_band = band(md, lookback_period, num_std, upper = False)

indicator_subplot = pq.Subplot([pq.TimeSeries('price', md.dates, md.c, color = 'blue'), 
                     pq.TimeSeries('sma', md.dates, mid_band, line_type = 'dotted', color = 'red'),
                     pq.TimeSeries('upper band', md.dates, upper_band, line_type = 'dashed', color = 'green'),
                     pq.TimeSeries('lower_band', md.dates, lower_band, line_type = 'dashed', color = 'green')],
                     ylabel = 'Indicators')
plot = pq.Plot(indicator_subplot, title = 'BTC')
plot.draw()

#cell 8
def bollinger_band_signal(md, ind): # md is MarketData, ind is indicator values, a dictionary of indicator values
    signal = np.where(md.h > np.nan_to_num(ind['upper_band']), 2, 0)
    signal = np.where(md.l < np.nan_to_num(ind['lower_band']), -2, signal)
    signal = np.where((md.h > np.nan_to_num(ind['mid_band'])) & (signal == 0), 1, signal) # price crossed above simple moving avg but not above upper band
    signal = np.where((md.l < np.nan_to_num(ind['mid_band'])) & (signal == 0), -1, signal) # price crossed below simple moving avg but not below lower band
    return signal

#cell 10
signal = bollinger_band_signal(md, {'upper_band' : upper_band, 'lower_band' : lower_band, 'mid_band' : mid_band})
signal_subplot = pq.Subplot([pq.TimeSeries('signal', md.dates, signal)], ylabel = 'Signal')
plot = pq.Plot([indicator_subplot, signal_subplot], title = 'BTC', show_date_gaps = False)
plot.draw()

#cell 12
plot = pq.Plot([indicator_subplot, signal_subplot], title = 'BTC', date_range = ('2018-03-24', '2018-03-25'))
plot.draw()

#cell 14
def bollinger_band_trading_rule(strategy, symbol, i, date, marketdata, indicator_values, signal_values, account):
    curr_pos = account.position(symbol, date)
    signal_value = signal_values[i]
    risk_percent = 0.1
    
    # if we don't already have a position, check if we should enter a trade
    if math.isclose(curr_pos, 0):
        if signal_value == 2 or signal_value == -2:
            curr_equity = account.equity(date)
            order_qty = np.round(curr_equity * risk_percent / marketdata.c[i] * np.sign(signal_value))
            trigger_price = marketdata.c[i]
            reason_code = pq.ReasonCode.ENTER_LONG if order_qty > 0 else pq.ReasonCode.ENTER_SHORT

            return [pq.StopLimitOrder(symbol, date, order_qty, trigger_price, reason_code = reason_code)]
    else: # We have a current position, so check if we should exit
        if (curr_pos > 0 and signal_value == -1) or (curr_pos < 0 and signal_value == 1):
            order_qty = -curr_pos
            reason_code = pq.ReasonCode.EXIT_LONG if order_qty < 0 else pq.ReasonCode.EXIT_SHORT
            return [pq.MarketOrder(symbol, date, order_qty, reason_code = reason_code)]
    return []


#cell 16
def market_simulator(strategy, orders, i, date, md):
    trades = []
    
    o, h, l, c = md.o[i], md.h[i], md.l[i], md.c[i]
    
    for order in orders:
        trade_price = np.nan
        
        if isinstance(order, pq.MarketOrder):
            trade_price = 0.5 * (o + h) if order.qty > 0 else 0.5 * (o + l)
        elif isinstance(order, pq.StopLimitOrder):
            if (order.qty > 0 and h > order.trigger_price) or (order.qty < 0 and l < order.trigger_price): # A stop order
                trade_price = 0.5 * (order.trigger_price + h) if order.qty > 0 else 0.5 * (order.trigger_price + l)
        else:
            raise Exception(f'unexpected order type: {order}')
            
        if np.isnan(trade_price): continue
            
        trade = pq.Trade(order.symbol, date, order.qty, trade_price, order = order, commission = 0, fee = 0)
        order.status = 'filled'
                           
        trades.append(trade)
                           
    return trades

#cell 18
contract = pq.Contract('BTC', multiplier = 1., marketdata = md)
strategy = pq.Strategy([contract])

# since pqstrat expects the indicator to take one argument, market data, and our sma function takes 2 arguments, we wrap it in a lambda 
strategy.add_indicator('mid_band', lambda md : sma(md, lookback_period))
strategy.add_indicator('upper_band', lambda md: band(md, lookback_period, num_std, upper = True))
strategy.add_indicator('lower_band', lambda md: band(md, lookback_period, num_std, upper = False))
strategy.add_signal('bb_signal', bollinger_band_signal)

# ask pqstrat to call our trading rule when the signal has one of the values [-2, -1, 1, 2]
strategy.add_rule('bb_trading_rule', bollinger_band_trading_rule, 
                  signal_name = 'bb_signal', sig_true_values = [-2, -1, 1, 2])

strategy.add_market_sim(market_simulator, symbols = ['BTC'])

portfolio = pq.Portfolio()
portfolio.add_strategy('bb_strategy', strategy)
portfolio.run()

#cell 20
strategy.plot()

#cell 21
strategy.df_returns()

#cell 22
strategy.evaluate_returns();

#cell 24
from pyqstrat.examples.build_example_strategy import build_example_strategy

strategy1 = build_example_strategy(lookback_period = 5, num_std = 1)
strategy2 = build_example_strategy(lookback_period = 10, num_std = 2)
portfolio = pq.Portfolio()
portfolio.add_strategy('bb_strategy_1', strategy1)
portfolio.add_strategy('bb_strategy_2', strategy2)
portfolio.run()

#cell 26
strategy1.plot(title = 'Strategy 1')

#cell 27
strategy2.plot(title = 'Strategy 2')

#cell 28
portfolio.plot()

#cell 29


