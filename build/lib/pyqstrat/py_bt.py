#cell 0
import numpy as np
import pandas as pd
import sys
from collections import defaultdict
from pprint import pformat
from collections import deque
import math
from functools import reduce
from scipy.ndimage.interpolation import shift

import pandas as pd
from copy import copy
import numpy as np
from pybt.plot import *
from pybt.pybt_utils import *
from pybt.marketdata import *
from pybt.evaluator import Evaluator, compute_return_metrics, plot_return_metrics, display_return_metrics

#TODO:  

# 1.  Add rebalance rule to portfolio
# 2.  Add plotting to portfolio
# 3.  Add currency to add_symbol function
# 4.  Change add_symbol to add_instrument
# 5.  Add timer
# 6.  Additional order types:
#      a.  stop trail limit (trigger price updated as current market price changes).
#      b.  market and limit on close
# 7.  Add target order type??
# 8.  Add ability to use Bid / Offer data instead of OHLC
# 9.  Benchmarks - Return and Correlation
# 10.  Commissions & Fees
#      a.  Fixed commision per future or stock
#      b.  Per ticket fees
#      c.  Short interest
#      d.  ETF Fees 
# 11.  Dynamically selecting universe of symbols

#cell 1
class Portfolio:
    def __init__(self, name = 'main'):
        self.name = name
        self.strategies = {}
        
    def add_strategy(self, name, strategy):
        self.strategies[name] = strategy
        strategy.portfolio = self
        strategy.name = name
        
    def run_indicators(self, strategy_names = None):
        if strategy_names is None: strategy_names = self.strategies.keys()
        for name in strategy_names: self.strategies[name].run_indicators()
                
    def run_signals(self, strategy_names = None):
        if strategy_names is None: strategy_names = self.strategies.keys()
        for name in strategy_names: self.strategies[name].run_signals()
                
    def run_rules(self, strategy_names = None, start_date = None, end_date = None, run_first = False, run_last = True):
        start_date, end_date = str2date(start_date), str2date(end_date)
        if strategy_names is None: strategy_names = list(self.strategies.keys())
        strategies = [self.strategies[key] for key in strategy_names]
        
        min_date = min([strategy.dates[0] for strategy in strategies])
        if start_date: min_date = max(min_date, start_date)
        max_date = max([strategy.dates[-1] for strategy in strategies])
        if end_date: max_date = min(max_date, end_date)
        
        iter_list = []
        
        for strategy in strategies:
            dates, iterations = strategy._get_iteration_indices(start_date = start_date, end_date = end_date, run_first = run_first, run_last = run_last)
            iter_list.append((strategy, dates, iterations))
            
        dates_list = [tup[1] for tup in iter_list]
        
        #for v in self.rebalance_rules.values():
        #    _, freq = v
        #    rebalance_dates = pd.date_range(min_date, max_date, freq = freq).values
        #    dates_list.append(rebalance_dates)
        
        all_dates = np.array(reduce(np.union1d, dates_list))

        iterations = [[] for x in range(len(all_dates))]

        for tup in iter_list: # per strategy
            strategy = tup[0]
            dates = tup[1]
            iter_tup = tup[2] # vector with list of (rule, symbol, iter_params dict)
            for i, date in enumerate(dates):
                idx = np.searchsorted(all_dates, date)
                iterations[idx].append((Strategy._iterate, (strategy, i, iter_tup[i])))
                
        #for name, tup in self.rebalance_rules.items():
        #    rule, freq = tup
        #    rebalance_dates = pd.date_range(all_dates[0], all_dates[-1], freq = freq).values
        #    rebalance_indices = np.where(np.in1d(all_dates, rebalance_dates))[0]
        #    iterations[idx].append(lambda : Portfolio.rebalance(rule), idx)
                 
        self.iterations = iterations # For debugging
                
        for iter_idx, tup_list in enumerate(iterations):
            for tup in tup_list:
                func = tup[0]
                args = tup[1]
                func(*args)
                
    def run(self, strategy_names = None, start_date = None, end_date = None, run_first = False, run_last = True):
        start_date, end_date = str2date(start_date), str2date(end_date)
        self.run_indicators()
        self.run_signals()
        return self.run_rules(strategy_names, start_date, end_date, run_first, run_last)
        
    def returns_df(self, sampling_frequency = 'D', strategy_names = None):
        if strategy_names is None: strategy_names = self.strategies.keys()
        equity_list = []
        for name in strategy_names:
            equity = self.strategies[name].returns(sampling_frequency = sampling_frequency)[['equity']]
            equity.columns = [name]
            equity_list.append(equity)
        ret_df = pd.concat(equity_list, axis = 1)
        ret_df['equity'] = ret_df.sum(axis = 1)
        ret_df['ret'] = ret_df.equity.pct_change()
        return ret_df
        
    def evaluate_returns(self, sampling_frequency = 'D', strategy_names = None, plot = True, float_precision = 4):
        returns = self.returns_df(sampling_freq, strategy_names)
        ev = compute_return_metrics(returns.index.values, returns.ret.values, returns.equity.values[0])
        display_return_metrics(ev.metrics(), float_precision = float_precision)
        if plot: plot_return_metrics(ev.metrics())
        return ev.metrics()
    
    def plot_returns(self, sampling_frequency = 'D', strategy_names = None):
        returns = self.returns_df(sampling_frequency, strategy_names)
        ev = compute_return_metrics(returns.index.values, returns.ret.values, returns.equity.values[0])
        plot_return_metrics(ev.metrics())
        
    def __repr__(self):
        return '{0} {1} {2}'.format(self.name, pformat(self.strategies.keys()))
        
class MarketOrder:
    def __init__(self, symbol, date, qty, reason_code = None, status = 'open'):
        self.symbol = symbol
        self.date = date
        self.qty = qty
        self.reason_code = reason_code
        self.status = status
        
    def __repr__(self):
        return f'{self.symbol} {pd.Timestamp(self.date).to_pydatetime():%Y-%m-%d %H:%M} qty: {self.qty}' + (f' {self.reason_code}' if self.reason_code else '') + f' {self.status}'
    
    def params(self):
        return {}
        
class LimitOrder:
    def __init__(self, symbol, date, qty, limit_price, reason_code = None, status = 'open'):
        self.symbol = symbol
        self.date = date
        self.qty = qty
        self.reason_code = reason_code
        self.limit_price = limit_price
        self.status = status
        
    def __repr__(self):
        return f'{self.symbol} {pd.Timestamp(self.date).to_pydatetime():%Y-%m-%d %H:%M} qty: {self.qty} lmt_prc: {self.limit_price}' + (
            f' {self.reason_code}'  if self.reason_code else '') + f' {self.status}'
    
    def params(self):
        return {'limit_price' : self.limit_price}
    
class RollOrder:
    '''
    Used to roll futures
    '''
    def __init__(self, symbol, date, close_qty, reopen_qty, reason_code = 'RL', status = 'open'):
        self.symbol = symbol
        self.date = date
        self.close_qty = close_qty
        self.reopen_qty = reopen_qty
        self.reason_code = reason_code
        self.qty = close_qty # For display purposes when we print varying order types
        self.status = status
        
    def params(self):
        return {'close_qty' : self.close_qty, 'reopen_qty' : self.reopen_qty}
        
    def __repr__(self):
        return f'{self.symbol} {pd.Timestamp(self.date).to_pydatetime():%Y-%m-%d %H:%M} close_qty: {self.close_qty} reopen_qty: {self.reopen_qty}' + (
            f' {self.reason_code}' if self.reason_code else '') + f' {self.status}'
  
class StopLimitOrder:
    '''
    Triggered when trigger price is exceeded.  Becomes either a market or limit order at that point
    '''
    def __init__(self, symbol, date, qty, trigger_price, limit_price = np.nan, reason_code = None, status = 'open'):
        self.symbol = symbol
        self.date = date
        self.qty = qty
        self.trigger_price = trigger_price
        self.limit_price = limit_price
        self.reason_code = reason_code
        self.triggered = False
        self.status =  status
        
    def params(self):
        return {'trigger_price' : self.trigger_price, 'limit_price' : self.limit_price}
        
    def __repr__(self):
        return f'{self.symbol} {pd.Timestamp(self.date).to_pydatetime():%Y-%m-%d %H:%M} qty: {self.qty} trigger_prc: {self.trigger_price} limit_prc: {self.limit_price}' + (
            f' {self.reason_code}' if self.reason_code else '') + f' {self.status}'
                
class Trade:
    def __init__(self, symbol, date, qty, price, fee = 0., commission = 0., order = None):
        self.symbol = symbol
        self.date = date
        self.qty = qty
        self.price = price
        self.fee = fee
        self.commission = commission
        self.order = order
        
    def __repr__(self):
        return '{} {:%Y-%m-%d %H:%M} qty: {} prc: {}{}{} order: {}'.format(self.symbol, pd.Timestamp(self.date).to_pydatetime(), self.qty, self.price, 
                                                  ' ' + str(self.fee) if self.fee != 0 else '', 
                                                  ' ' + str(self.commission) if self.commission != 0 else '', 
                                                  self.order)

def _calc_pnl(open_trades, new_trades, ending_close, multiplier):
    '''
    >>> from collections import deque
    >>> trades = deque([Trade('IBM', np.datetime64('2018-01-01 10:15:00'), 3, 51.),
    ...              Trade('IBM', np.datetime64('2018-01-01 10:20:00'), 10, 50.),
    ...              Trade('IBM', np.datetime64('2018-01-02 11:20:00'), -5, 45.)])

    >>> print(calc_pnl(open_trades = deque(), trades = trades, ending_close = 54, multiplier = 100))
    (deque([IBM 2018-01-01T10:20:00 8 50.0 None]), 3200.0, -2800.0)
    >>> trades = deque([Trade('IBM', np.datetime64('2018-01-01 10:15:00'), -8, 10.),
    ...          Trade('IBM', np.datetime64('2018-01-01 10:20:00'), 9, 11.),
    ...          Trade('IBM', np.datetime64('2018-01-02 11:20:00'), -4, 6.)])
    >>> print(calc_pnl(open_trades = deque(), trades = trades, ending_close = 5.8, multiplier = 100))
    (deque([IBM 2018-01-02T11:20:00 -3 6.0 None]), 60.00000000000006, -1300.0)
    '''
    realized = 0.
    unrealized = 0.
    
    trades = copy(new_trades)
    
    while (len(trades)):
        trade = trades[0]
        if not len(open_trades) or (np.sign(open_trades[-1].qty) == np.sign(trade.qty)):
            open_trades.append(copy(trade))
            trades.popleft()
            continue
            
        if abs(trade.qty) > abs(open_trades[0].qty):
            open_trade = open_trades.popleft()
            realized += open_trade.qty * multiplier * (trade.price - open_trade.price)
            trade.qty += open_trade.qty
        else:
            open_trade = open_trades[0]
            realized += trade.qty * multiplier * (open_trades[-1].price - trade.price)
            trades.popleft()
            open_trade.qty += trade.qty
 
    unrealized = sum([open_trade.qty * (ending_close - open_trade.price) for open_trade in open_trades]) * multiplier

    return open_trades, unrealized, realized
        
class SymbolPNL:
    def __init__(self, symbol, multiplier, marketdata):
        self.symbol = symbol
        self.multiplier = multiplier
        self.marketdata = marketdata
        self.dates = marketdata.dates
        self.unrealized = np.empty(len(self.dates), dtype = np.float) * np.nan; self.unrealized[0] = 0
        self.realized = np.empty(len(self.dates), dtype = np.float) * np.nan; self.realized[0] = 0
        
        #TODO: Add commission and fee from trades
        self.commission = np.empty(len(self.dates), dtype = np.float) * 0; self.commission[0] = 0
        self.fee = np.empty(len(self.dates), dtype = np.float) * 0; self.fee[0] = 0
        
        self.net_pnl = np.empty(len(self.dates), dtype = np.float) * np.nan; self.net_pnl[0] = 0
        self.position = np.empty(len(self.dates), dtype = np.float) * np.nan; self.position[0] = 0
        self.close = marketdata.c
        self._trades = []
        self.open_trades = deque()
        
    def add_trades(self, trades):
        self._trades += trades
        
    def calc(self, prev_i, i):
        calc_trades = deque([trade for trade in self._trades if trade.date > self.dates[prev_i] and trade.date <= self.dates[i]])
        #trade = self._trades[0]
        
        if not np.isfinite(self.close[i]):
            unrealized = self.unrealized[prev_i]
            realized = 0. 
        else:
            open_trades, unrealized, realized = _calc_pnl(self.open_trades, calc_trades, self.close[i], self.multiplier)
            self.open_trades = open_trades
            
        self.unrealized[i] = unrealized
        self.realized[i] = self.realized[prev_i] + realized
        self.position[i] = self.position[prev_i] + sum([trade.qty for trade in calc_trades])
        self.net_pnl[i] = self.realized[i] + self.unrealized[i] - self.commission[i] - self.fee[i]
        
    def trades(self, start_date = None, end_date = None):
        start_date, end_date = str2date(start_date), str2date(end_date)
        trades = [trade for trade in self._trades if (start_date is None or trade.date >= start_date) and (end_date is None or trade.date <= end_date)]
        return trades
         
    def to_df(self):
        df = pd.DataFrame({'date' : self.dates, 'unrealized' : self.unrealized, 'realized' : self.realized, 
                           'fee' : self.fee, 'net_pnl' : self.net_pnl, 'position' : self.position})
        df.dropna(subset = ['unrealized', 'realized'], inplace = True)
        df['symbol'] = self.symbol
        return df[['symbol', 'date', 'unrealized', 'realized', 'fee', 'net_pnl', 'position']].set_index('date')
    
class Symbol:
    def __init__(self, symbol, multiplier, marketdata):
        self.symbol = symbol
        self.multiplier = multiplier
        self.marketdata = marketdata

class Account:
    def __init__(self, dates, symbols, starting_equity = 1.0e6, calc_frequency = 'D'):
        self.calc_freq = calc_frequency
        self.symbol_pnls = defaultdict()
        self.all_dates = dates
        
        if calc_frequency == 'D': 
            calc_dates = dates.astype('M8[D]')
        else: 
            raise Exception('unknown calc frequency: {}'.format(calc_frequency))
        self.calc_dates = np.unique(calc_dates)
        self.calc_indices = np.searchsorted(dates, self.calc_dates, side='left') - 1
        if self.calc_indices[0] == -1: self.calc_indices[0] = 0
        self.current_calc_index = 0
        self._equity = np.empty(len(dates), np.float) * np.nan; 
        self._equity[0] = starting_equity
        self.marketdata = {}
        
        for symbol in symbols: self.add_symbol(symbol.symbol, symbol.multiplier, symbol.marketdata)
            
    def starting_equity(self):
        return self._equity[0]
        
    def symbols(self):
        return list(self.symbol_pnls.keys())
        
    def add_symbol(self, symbol, multiplier, marketdata):
        #TODO: All market data must be aligned so we can use last calc index, etc.
        self.symbol_pnls[symbol] = SymbolPNL(symbol, multiplier, marketdata)
        self.marketdata[symbol] = marketdata
        
    def _add_trades(self, symbol, trades):
        self.symbol_pnls[symbol].add_trades(trades)
        
    def calc(self, i):
        calc_indices = self.calc_indices[:]
        if self.current_calc_index == i: return
        intermediate_calc_indices = np.ravel(np.where(np.logical_and(calc_indices > self.current_calc_index, calc_indices <= i)))

        if not len(intermediate_calc_indices) or calc_indices[intermediate_calc_indices[-1]] != i: 
            calc_indices = np.append(calc_indices, i)
            intermediate_calc_indices = np.append(intermediate_calc_indices, len(calc_indices) - 1)
            
        for symbol, symbol_pnl in self.symbol_pnls.items():
            prev_calc_index = self.current_calc_index
            for idx in intermediate_calc_indices:
                calc_index = calc_indices[idx]
                symbol_pnl.calc(prev_calc_index, calc_index)
                self._equity[calc_index] = self._equity[prev_calc_index] + symbol_pnl.net_pnl[calc_index] - symbol_pnl.net_pnl[prev_calc_index]
                prev_calc_index = calc_index
                
        self.current_calc_index = i
        
    def position(self, symbol, date):
        i = self.find_index_before(date)
        self.calc(i)
        return self.symbol_pnls[symbol].position[i]
    
    def equity(self, date):
        i = self.find_index_before(date)
        self.calc(i)
        return self._equity[i]
    
    def trades(self, symbol = None, start_date = None, end_date = None):
        start_date, end_date = str2date(start_date), str2date(end_date)
        if symbol is None:
            trades = []
            for symbol, sym_pnl in self.symbol_pnls:
                trades += sym_pnl.trades(start_date, end_date)
            return trades
        else:
            return self.symbol_pnls[symbol].trades(start_date, end_date)
        
    def find_index_before(self, date):
        return np.searchsorted(self.all_dates, date)
        
    def transfer_cash(self, date, amount):
        '''Move cash from one portfolio to another'''
        i = self.find_index_before(date)
        curr_equity = self.equity(date)
        if (amount > curr_equity): amount = curr_equity # Cannot make equity negative
        self._equity[i] -= amount
        return amount

    def pnl_df(self, symbol = None):
        if symbol:
            ret = self.symbol_pnls[symbol].to_df()
        else:
            dfs = []
            for symbol, symbol_pnl in self.symbol_pnls.items():
                df = symbol_pnl.to_df()
                dfs.append(df)
            ret = pd.concat(dfs)
            ret = ret.reset_index().groupby('date').sum()
        equity_df = pd.DataFrame({'equity' : self._equity}, index = self.all_dates).dropna()
        ret = pd.merge(ret, equity_df, left_index = True, right_index = True, how = 'outer')
        ret.index.name = 'date'
        return ret
    
    def trades_df(self, symbol = None, start_date = None, end_date = None):
        start_date, end_date = str2date(start_date), str2date(end_date)
        if symbol:
            trades = self.symbol_pnls[symbol].trades(start_date, end_date)
        else:
            trades = [v.trades(start_date, end_date) for v in self.symbol_pnls.values()]
            trades = [trade for sublist in trades for trade in sublist] # flatten list
        trades_df = pd.DataFrame.from_records([(trade.symbol, trade.date, trade.qty, trade.price, trade.fee, trade.commission, trade.order.date, trade.order.qty, trade.order.params()) for trade in trades],
                    columns = ['symbol', 'date', 'qty', 'price', 'fee', 'commission', 'order_date', 'order_qty', 'order_params'])
        return trades_df
            
class Strategy:
    def __init__(self, account):
        self.name = None
        self.account = account
        self.symbols = self.account.symbols()
        self.indicators = {}
        self.indicator_values = defaultdict(dict)
        self.signals = {}
        self.signal_values = defaultdict(dict)
        self.rules = {}
        self.rule_signals = {}
        self.market_sims = {}
        self.trades = defaultdict(list)
        self._orders = []
        self.dates = self.account.all_dates
        
    def add_indicator(self, name, indicator):
        self.indicators[name] = indicator
        
    def add_signal(self, name, signal):
        self.signals[name] = signal
        
    def add_rule(self, name, rule, signal_name, sig_true_values):
        self.rule_signals[name] = (signal_name, sig_true_values)
        self.rules[name] = rule
        
    def add_market_sim(self, market_sim, symbols = None):
        if symbols is None: symbols = self.symbols
        for symbol in symbols: self.market_sims[symbol] = market_sim
        
    def run_indicators(self, indicator_names = None, symbols = None):
        if indicator_names is None: indicator_names = self.indicators.keys()
        if symbols is None: symbols = self.symbols
            
        for indicator_name in indicator_names:
            indicator = self.indicators[indicator_name]
            for symbol in symbols:
                marketdata = self.account.marketdata[symbol]
                self.indicator_values[symbol][indicator_name] = indicator(marketdata)
                
    def run_signals(self, signal_names = None, symbols = None):
        if signal_names is None: signal_names = self.signals.keys()
        if symbols is None: symbols = self.symbols
        
        for signal_name in signal_names:
            signal = self.signals[signal_name]
            for symbol in symbols:
                marketdata = self.account.marketdata[symbol]
                self.signal_values[symbol][signal_name] = signal(marketdata, self.indicator_values[symbol])
                
                
    def run_rules(self, rule_names = None, symbols = None, start_date = None, end_date = None, run_first = False, run_last = True):
        start_date, end_date = str2date(start_date), str2date(end_date)
        dates, iterations = self._get_iteration_indices(rule_names, symbols, start_date, end_date, run_first, run_last)
        # Now we know which rules, symbols need to be applied for each iteration, go through each iteration and apply them
        # in the same order they were added to the strategy
        for i, tup_list in enumerate(iterations):
            self._iterate(i, tup_list)
        
    def _get_iteration_indices(self, rule_names = None, symbols = None, start_date = None, end_date = None, 
                  run_first = False, run_last = True):
        '''
        >>> from minimock import Mock
        >>> strat = Mock('strategy', tracker = None)
        >>> rule1 = Mock('rule1', tracker = None)
        >>> rule1.signal.name = 'sig1'
        >>> rule1.__call__.mock_returns = [LimitOrder('IBM', 15, 10.)]
        >>> strat.rules = {'rule1' : rule1}
        >>> strat.orders = defaultdict(list)
        >>> strat.portfolio.marketdata = {'IBM' : pd.DataFrame({'sig1' : [1., 1., -1., 1]})}
        >>> market_sim = Mock('market_sim')
        >>> market_sim.__call__.mock_returns = [Trade('IBM', np.datetime64('2018-01-01'), 15, 9)]
        >>> strat.trades = defaultdict(list)
        >>> strat.market_sims = {'IBM' : market_sim}
        >>> strategy.run_rules(strat, rule_names = ['rule1'], symbols = ['IBM'])
        Called market_sim.__call__([IBM 15 10.0 open], 2)
        >>> strat.trades
        defaultdict(<class 'list'>, {'IBM': [IBM 2018-01-01 15 9 None]})
        '''
        start_date, end_date = str2date(start_date), str2date(end_date)
        if rule_names is None: rule_names = self.rules.keys()
        if symbols is None: symbols = self.symbols
            
        dates = self.dates
        num_dates = len(dates)
                    
        iterations = [[] for x in range(num_dates)]
        self.orders_iter = [[] for x in range(num_dates)]
            
        for rule_name in rule_names:
            rule = self.rules[rule_name]
            for symbol in symbols:
                marketdata = self.account.marketdata[symbol]
                market_sim = self.market_sims[symbol]
                signal_name = self.rule_signals[rule_name][0]
                sig_true_values = self.rule_signals[rule_name][1]
                sig_values = self.signal_values[symbol][signal_name]
                dates = marketdata.dates
                null_value = False if sig_values.dtype == np.dtype('bool') else np.nan
                if start_date: sig_values[0:np.searchsorted(dates, start_date)] = null_value
                if end_date:   sig_values[np.searchsorted(dates, end_date):] = null_value
                indices = np.nonzero(np.isin(sig_values, sig_true_values))[0]
                if indices[-1] == len(sig_values) -1: indices = indices[:-1] # Don't run rules on last index since we cannot fill any orders
                if run_first and indices[0] != 0: indices = np.insert(indices, 0, 0)
                if run_last and indices[-1] != len(sig_values) - 2: indices = np.append(indices, len(sig_values) - 2)
                indicator_values = self.indicator_values[symbol]
                iteration_params = {'market_sim' : market_sim, 'indicator_values' : indicator_values, 'signal_values' : sig_values, 'marketdata' : marketdata}
                for idx in indices: iterations[idx].append((rule, symbol, iteration_params))
                    
        self.iterations = iterations # For debugging
                    
        return self.dates, iterations
         
    def _iterate(self, i, tup_list):
        for tup in self.orders_iter[i]:
            try:
                open_orders, symbol, params = tup
                open_orders = self._sim_market(i, open_orders, symbol, params)
                if len(open_orders): self.orders_iter[i + 1].append((open_orders, symbol, params))
            except Exception as e:
                raise type(e)(f'Exception: {str(e)} at rule: {type(tup[0])} symbol: {tup[1]} index: {i}').with_traceback(sys.exc_info()[2])
                
        for tup in tup_list:
            try:
                rule, symbol, params = tup
                open_orders = self._get_orders(i, rule, symbol, params)
                self._orders += open_orders
                if len(open_orders): self.orders_iter[i + 1].append((open_orders, symbol, params))
            except Exception as e:
                raise type(e)(f'Exception: {str(e)} at rule: {type(tup[0])} symbol: {tup[1]} index: {i}').with_traceback(sys.exc_info()[2])
                    
    def _get_orders(self, idx, rule, symbol, params):
        indicator_values, signal_values, marketdata = (params['indicator_values'], params['signal_values'], params['marketdata'])
        open_orders = rule(self, symbol, idx, self.dates[idx], marketdata, indicator_values, signal_values, self.account)
        return open_orders
        
    def _sim_market(self, idx, open_orders, symbol, params):
        '''
        Keep iterating while we have open orders since they may get filled
        TODO: For limit orders and trigger orders we can be smarter here and reduce indices like quantstrat does
        '''
        market_sim = params['market_sim']
        trades = market_sim(self, open_orders, idx, self.dates[idx], self.account.marketdata[symbol])
        if len(trades) == 0: return[]
        self.trades[symbol] += trades
        self.account._add_trades(symbol, trades)
        self.account.calc(idx)
        open_orders = [order for order in open_orders if order.status == 'open']
        return open_orders
                        

            
    def data(self, symbols = None, add_pnl_df = True, start_date = None, end_date = None):
        '''
        Add indicators and signals to end of market data and return as a dataframe
        '''
        start_date, end_date = str2date(start_date), str2date(end_date)
        if symbols is None: symbols = self.symbols
        if not isinstance(symbols, list): symbols = [symbols]
            
        mds = []
            
        for symbol in symbols:
            md = self.account.marketdata[symbol].to_df(start_date, end_date)
            
            md.insert(0, 'symbol', symbol)
            if add_pnl_df: 
                pnl_df = self.account.pnl_df(symbol)
                del pnl_df['symbol']

            indicator_values = self.indicator_values[symbol]

            for k in sorted(indicator_values.keys()):
                name = k
                if name in md.columns: name = name + '.ind' # if we have a market data column with the same name as the indicator
                md.insert(len(md.columns), name, indicator_values[k])

            signal_values = self.signal_values[symbol]

            for k in sorted(signal_values.keys()):
                name = k
                if name in md.columns: name = name + '.sig'
                md.insert(len(md.columns), name, signal_values[k])
            
            if add_pnl_df: md = pd.merge(md, pnl_df, left_index = True, right_index = True, how = 'left')
            # Add counter column for debugging
            md.insert(len(md.columns), 'i', np.arange(len(md)))
            
            mds.append(md)
            
        return pd.concat(mds)
    
    def marketdata(self, symbol):
        return self.account.marketdata[symbol]
    
    def trades(self, symbol = None, start_date = None, end_date = None):
        start_date, end_date = str2date(start_date), str2date(end_date)
        return self.account.trades(symbol, start_date, end_date)
    
    def trades_df(self, symbol = None, start_date = None, end_date = None):
        start_date, end_date = str2date(start_date), str2date(end_date)
        return self.account.trades_df(symbol, start_date, end_date)
    
    def orders(self, symbol = None, start_date = None, end_date = None):
        start_date, end_date = str2date(start_date), str2date(end_date)
        return [order for order in self._orders if (symbol is None or order.symbol == symbol) and (
            start_date is None or order.date >= start_date) and (end_date is None or order.date <= end_date)]
    
    def orders_df(self, symbol = None, start_date = None, end_date = None):
        start_date, end_date = str2date(start_date), str2date(end_date)
        orders = self.orders(symbol, start_date, end_date)
        orders_df = pd.DataFrame.from_records([(order.symbol, order.date, order.qty, str(order.params)) for order in orders], columns = ['symbol', 'date', 'qty', 'params'])
        return orders_df
   
    def pnl_df(self, symbol = None):
        return self.account.pnl_df(symbol)
    
    def returns(self, symbol = None, sampling_frequency = 'D'):
        pnl = self.pnl_df(symbol)[['equity']]
        pnl.equity = pnl.equity.ffill()
        pnl = pnl.resample(sampling_frequency).last()
        pnl['ret'] = pnl.equity.pct_change()
        return pnl
    
    def plot(self, symbols = None, md_columns = 'ohlc', pnl_columns = 'equity', title = None, figsize = (20, 15), date_range = None, 
             date_format = None, sampling_frequency = None):
        date_range = strtup2date(date_range)
        if symbols is None: symbols = self.symbols
        if not isinstance(symbols, list): symbols = [symbols]
        if not isinstance(md_columns, list): md_columns = [md_columns]
        if not isinstance(pnl_columns, list): pnl_columns = [pnl_columns]
        for symbol in symbols:
            md = self.marketdata(symbol)
            md_dates = md.dates
            if md_columns == ['ohlc']:
                md_list = [OHLC('price', dates = md_dates, o = md.o, h = md.h, l = md.l, c = md.c, v = md.v)]
            else:
                md_list = [TimeSeries(md_column, dates = md_dates, values = getattr(md, md_column)) for md_column in md_columns]
            indicator_list = [TimeSeries(indicator_name, dates = md_dates, values = self.indicator_values[symbol][indicator_name], line_type = '--'
                                        ) for indicator_name in self.indicators.keys() if indicator_name in self.indicator_values[symbol]]
            signal_list = [TimeSeries(signal_name, dates = md_dates, values = self.signal_values[symbol][signal_name]
                                     ) for signal_name in self.signals.keys() if signal_name in self.signal_values[symbol]]
            _pnl_df = self.pnl_df(symbol)
            pnl_list = [TimeSeries(pnl_column, dates = _pnl_df.index.values, values = _pnl_df[pnl_column].values) for pnl_column in pnl_columns]
            positions = (_pnl_df.index.values, _pnl_df.position.values)
            main_subplot = Subplot(indicator_list +  md_list + get_long_short_trade_sets(self.trades[symbol], positions), height_ratio = 0.5, title = 'Indicators')
            signal_subplot = Subplot(signal_list, title = 'Trend', height_ratio = 0.167)
            pnl_subplot = Subplot(pnl_list, title = 'Equity', height_ratio = 0.167, log_y = True, y_tick_format = '${x:,.0f}')
            position = _pnl_df.position.values
            pos_subplot = Subplot([TimeSeries('position', dates = _pnl_df.index.values, values = position, plot_type = 'filled_line')], title = 'Position', height_ratio = 0.167)
            plot = Plot([main_subplot, signal_subplot, pos_subplot, pnl_subplot], figsize = figsize,
                                date_range = date_range, date_format = date_format, sampling_frequency = sampling_frequency, title = title)
            plot.draw()
            
    def evaluate_returns(self, symbol = None, plot = True, float_precision = 4):
        returns = self.returns(symbol)
        ev = compute_return_metrics(returns.index.values, returns.ret.values, self.account.starting_equity())
        display_return_metrics(ev.metrics(), float_precision = float_precision)
        if plot: plot_return_metrics(ev.metrics())
        return ev.metrics()
    
    def plot_returns(self, symbol = None):
        returns = self.returns(symbol)
        ev = compute_return_metrics(returns.index.values, returns.ret.values, self.account.starting_equity())
        plot_return_metrics(ev.metrics())
       
    def __repr__(self):
        return f'{pformat(self.indicators)} {pformat(self.rules)} {pformat(self.account)}'

if __name__ == '__main__':
    
    from datetime import datetime, timedelta
    
    set_defaults()

    def sim_order(order, i, date, md):
        symbol = order.symbol
        trade_price = np.nan
        skid_fraction = 0.5
        
        if not md.valid_row(i): return None
        
        if isinstance(order, MarketOrder):
            if order.qty > 0:
                trade_price = 0.5 * (md.c[i] + md.h[i])
            else:
                trade_price = 0.5 * (md.c[i] + md.l[i])
        elif order.qty > 0 and md.h[i] > order.trigger_price:
            trade_price = skid_fraction * max(md.o[i], order.trigger_price, md.l[i]) + (1 - skid_fraction) * md.h[i]
        elif order.qty < 0 and md.l[i] < order.trigger_price:
            trade_price = skid_fraction * min(md.o[i], order.trigger_price, md.h[i]) + (1 - skid_fraction) * md.l[i]
        else:
            return None
        sim_trade = Trade(symbol, date, order.qty, trade_price, order = order)
        return sim_trade
    
    def market_simulator(strategy, orders, i, date, marketdata):
        trades = []
        for order in orders:
            sim_trade = sim_order(order, i, date, marketdata)
            if sim_trade is None: continue
            if math.isclose(sim_trade.qty, order.qty): order.status = 'filled'
            trades.append(sim_trade)
        return trades
    
    def trade_rule(strategy, symbol, i, date, marketdata, indicator_values, signal_values, account):
        heat = 0.05
        
        if not marketdata.valid_row(i): return []
        
        curr_pos = account.position(symbol, date)
        
        if i == len(marketdata.dates) - 2: # Last date so get out of position
            if not math.isclose(curr_pos, 0): 
                return [MarketOrder(symbol, date, -curr_pos, reason_code = 'last_date')]
            else:
                return []
            
        trend = signal_values[i]
        fast_resistance, fast_support, slow_resistance, slow_support = (indicator_values['fast_resistance'][i], 
                indicator_values['fast_support'][i], indicator_values['slow_resistance'][i], indicator_values['slow_support'][i])
        
        if trend == 1:
            entry_limit = fast_resistance
            stop_limit = fast_support
        elif trend == -1:
            entry_limit = fast_support
            stop_limit = fast_resistance
        else:
            return []

        if math.isclose(curr_pos, 0): # We got a trade in the previous bar so put in a stop limit order
            if math.isclose(entry_limit, stop_limit): return []
            curr_equity = account.equity(date)
            order_qty = curr_equity * heat / (entry_limit - stop_limit)
            trigger_price = entry_limit
        else:
            order_qty = -curr_pos
            trigger_price = stop_limit
        
        order_qty = round(order_qty)
            
        if math.isclose(order_qty, 0): return []
        
        order = StopLimitOrder(symbol, date, order_qty, trigger_price)
        return [order]
    
    def get_support(lows, n): return pd.Series(lows).rolling(window = n, min_periods = 1).min().values

    def get_resistance(highs, n): return pd.Series(highs).rolling(window = n, min_periods = 1).max().values
    
    def get_trend(md, ind):
        trend = pd.Series(np.where(pd.Series(md.h) > shift(ind['slow_resistance'], 1, cval = np.nan), 1, 
                          np.where(pd.Series(md.l) < shift(ind['slow_support'], 1, cval = np.nan), -1, 
                          np.nan)))
        trend.fillna(method = 'ffill', inplace = True)
        return trend.values
    
    def build_strategy(account, fast_interval, slow_interval):
        strategy = Strategy(account)
        strategy.add_indicator('slow_resistance', lambda md : get_resistance(md.h, slow_interval))
        strategy.add_indicator('slow_support', lambda md : get_support(md.l, slow_interval))
        strategy.add_indicator('fast_resistance', lambda md : get_resistance(md.h, fast_interval))
        strategy.add_indicator('fast_support', lambda md : get_support(md.l, fast_interval))
        strategy.add_signal('trend', get_trend)
        strategy.add_market_sim(market_simulator)
        strategy.add_rule('trade_rule', trade_rule, 'trend', np.array([-1, 1]))
        return strategy
    
    np.random.seed(0)
    dates = np.arange(datetime(2018, 1, 1, 9, 0, 0), datetime(2018, 3, 1, 16, 0, 0), timedelta(minutes = 5))
    dates = np.array([dt for dt in dates.astype(object) if dt.hour >= 9 and dt.hour <= 16]).astype('M8[m]')
    rets = np.random.normal(size = len(dates)) / 1000
    c_0 = 100
    c = np.round(c_0 * np.cumprod(1 + rets), 2)
    l = np.round(c * (1. - np.abs(np.random.random(size = len(dates)) / 1000.)), 2)
    h = np.round(c * (1. + np.abs(np.random.random(size = len(dates)) / 1000.)), 2)
    o = np.round(l + (h - l) * np.random.random(size = len(dates)), 2)
    v = np.round(np.random.normal(size = len(dates)) * 100)
    
    portfolio = Portfolio()
    
    slow_interval = 0
    
    for days in [0.5, 1, 2]:
        # 1 day from 9 - 4 pm has 7 hours which translate to 7 x 12 = 84 5 minute periods
        fast_interval = round(days * 84) 
        slow_interval = round(5 * fast_interval)
        account = Account(dates, symbols = [Symbol('IBM', 1, MarketData(dates, c, o, h, l, v))])
        portfolio.add_strategy(f'strat_{days}', build_strategy(account, fast_interval, slow_interval))
    
    # Start at max slow days so all strategies start at the same time
    print('running')
    portfolio.run(start_date = dates[slow_interval])
    print('done')
    
    strat1 = portfolio.strategies['strat_0.5']
    #strat1.evaluate_returns();
    #strat1.plot(date_range = (np.datetime64('2018-01-08'), np.datetime64('2018-01-15')));
    #for name, strategy in portfolio.strategies.items():
    #    strategy.plot(title = name);

#cell 2
portfolio.plot_returns()

#cell 3


