#cell 2
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed") # ignore pesky warning, see https://github.com/numpy/numpy/pull/432

import pandas as pd
import numpy as np
import pyqstrat as pq
from pyqstrat.examples.build_example_strategy import build_example_strategy


strategy = build_example_strategy(lookback_period = 10, num_std = 2)



#cell 3
portfolio = pq.Portfolio()
portfolio.add_strategy('bb_strategy', strategy)
portfolio.run()

#cell 5
strategy.df_data().head()

#cell 7
strategy.df_pnl().head()

#cell 8
strategy.marketdata('BTC').df().head()

#cell 10
strategy.df_orders().head()

#cell 11
strategy.df_trades().head()

#cell 13
portfolio.df_returns()

#cell 15
strategy.trades(start_date = '2018-03-23', end_date = '2018-03-24')

#cell 17
strategy.evaluate_returns(plot = False);

#cell 19
def compute_num_long_trades(trades):
    return len([trade for trade in trades if trade.order.reason_code == pq.ReasonCode.ENTER_LONG])

def compute_num_short_trades(trades):
    return len([trade for trade in trades if trade.order.reason_code == pq.ReasonCode.ENTER_SHORT])

evaluator = pq.Evaluator(initial_metrics = {'trades' : strategy.trades()})

evaluator.add_scalar_metric('num_long_trades', compute_num_long_trades, dependencies = ['trades'])
evaluator.add_scalar_metric('num_short_trades', compute_num_short_trades, dependencies = ['trades'])

evaluator.compute()

print('Long Trades: {} Short Trades: {}'.format(evaluator.metric('num_long_trades'), evaluator.metric('num_short_trades')))

#cell 21
import collections

trades = strategy.trades()

entry_trades = [trade for trade in trades if trade.order.reason_code == pq.ReasonCode.ENTER_LONG]
exit_trades = [trade for trade in trades if trade.order.reason_code == pq.ReasonCode.EXIT_LONG]

def compute_mae(entry_trades, exit_trades, marketdata):
    
    md = marketdata
    mae = np.empty(len(entry_trades)) * np.nan
    
    round_trip_pnl = np.empty(len(entry_trades)) * np.nan

    for i, entry in enumerate(entry_trades):
        if i == len(exit_trades): break
        exit = exit_trades[i]
        _round_trip_pnl = entry.qty * (exit.price - entry.price)
        running_price = md.c[(md.dates >= entry.date) & (md.dates <= exit.date)]
        running_pnl = entry.qty * (running_price - entry.price)
        _mae = -1 * min(_round_trip_pnl, np.min(running_pnl))
        _mae = _mae / entry.price # Get mae in % terms
        _mae = max(0, _mae) # If we have no drawdown for this trade, set it to 0
        mae[i] = _mae
        round_trip_pnl[i] = _round_trip_pnl / entry.price # Also store round trip pnl for this trade since we will have to plot it
    return mae, round_trip_pnl
        
def get_trades(trades, entry):
    rc = [pq.ReasonCode.ENTER_LONG, pq.ReasonCode.ENTER_SHORT] if entry else [pq.ReasonCode.EXIT_LONG, pq.ReasonCode.EXIT_SHORT]
    return [trade for trade in trades if trade.order.reason_code in rc]

evaluator = pq.Evaluator(initial_metrics = {'trades' : strategy.trades(), 'marketdata' : strategy.marketdata('BTC')})
evaluator.add_scalar_metric('entry_trades', lambda trades : get_trades(trades, True), dependencies=['trades'])
evaluator.add_scalar_metric('exit_trades', lambda trades : get_trades(trades, False), dependencies=['trades'])
evaluator.add_scalar_metric('mae', compute_mae, dependencies=['entry_trades', 'exit_trades', 'marketdata'])

evaluator.compute()



#cell 24
mae = evaluator.metric('mae')[0]
round_trip_pnl = evaluator.metric('mae')[1]

# Separate out positive trades from negative trades
round_trip_profit = round_trip_pnl[round_trip_pnl >= 0]
mae_profit = mae[round_trip_pnl >= 0]

round_trip_loss = round_trip_pnl[round_trip_pnl <= 0]
mae_loss = mae[round_trip_pnl <= 0]


subplot = pq.Subplot([
    pq.XYData('Profitable Trade', mae_profit, round_trip_profit, plot_type = 'scatter', marker = '^', marker_color = 'green'),
    pq.XYData('Losing Trade', mae_loss, -1 * round_trip_loss, plot_type = 'scatter', marker = 'v', marker_color = 'red')],
    horizontal_lines = [pq.HorizontalLine(y = 0, color = 'black')],
    vertical_lines = [pq.VerticalLine(x = 0, color = 'black')],
    xlabel = 'Drawdown in %', ylabel = 'Profit / Loss in %')

plot = pq.Plot([subplot])
plot.draw()

#cell 26
subplot = pq.Subplot([
    pq.XYData('Profitable Trade', mae_profit, round_trip_profit, plot_type = 'scatter', marker = '^', marker_color = 'green'),
    pq.XYData('Losing Trade', mae_loss, -1 * round_trip_loss, plot_type = 'scatter', marker = 'v', marker_color = 'red')],
    horizontal_lines = [pq.HorizontalLine(y = 0, color = 'black')],
    vertical_lines = [pq.VerticalLine(x = 0, color = 'black'), pq.VerticalLine(name = 'Stop Loss', x = 0.03, color = 'blue')],
    xlabel = 'Drawdown in %', ylabel = 'Profit / Loss in %')

plot = pq.Plot([subplot])
plot.draw()

