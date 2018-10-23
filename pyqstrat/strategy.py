
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed") # another bogus warning, see https://github.com/numpy/numpy/pull/432
import numpy as np
import pandas as pd
import sys
from collections import defaultdict
from pprint import pformat
from collections import deque
import math
from functools import reduce

import pandas as pd
from copy import copy
import numpy as np

from pyqstrat.pq_utils import *
from pyqstrat.marketdata import *
from pyqstrat.orders import *
from pyqstrat.plot import *
from pyqstrat.evaluator import *


# In[3]:


def _calc_pnl(open_trades, new_trades, ending_close, multiplier):
    '''
    >>> from collections import deque
    >>> trades = deque([Trade('IBM', np.datetime64('2018-01-01 10:15:00'), 3, 51.),
    ...              Trade('IBM', np.datetime64('2018-01-01 10:20:00'), 10, 50.),
    ...              Trade('IBM', np.datetime64('2018-01-02 11:20:00'), -5, 45.)])
    >>> print(_calc_pnl(open_trades = deque(), new_trades = trades, ending_close = 54, multiplier = 100))
    (deque([IBM 2018-01-01 10:20 qty: 8 prc: 50.0 order: None]), 3200.0, -2800.0)
    >>> trades = deque([Trade('IBM', np.datetime64('2018-01-01 10:15:00'), -8, 10.),
    ...          Trade('IBM', np.datetime64('2018-01-01 10:20:00'), 9, 11.),
    ...          Trade('IBM', np.datetime64('2018-01-02 11:20:00'), -4, 6.)])
    >>> print(_calc_pnl(open_trades = deque(), new_trades = trades, ending_close = 5.8, multiplier = 100))
    (deque([IBM 2018-01-02 11:20 qty: -3 prc: 6.0 order: None]), 60.00000000000006, -1300.0)
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

class Trade:
    def __init__(self, symbol, date, qty, price, fee = 0., commission = 0., order = None):
        '''Args:
            symbol: a string
            date: Trade execution datetime
            qty: Number of contracts or shares filled
            price: Trade price
            fee: Fees paid to brokers or others. Default 0
            commision: Commission paid to brokers or others. Default 0
            order: A reference to the order that created this trade. Default None
        '''
        assert(isinstance(symbol, str) and len(symbol) > 0)
        assert(np.isfinite(qty))
        assert(np.isfinite(price))
        assert(np.isfinite(fee))
        assert(np.isfinite(commission))
        
        self.symbol = symbol
        self.date = date
        self.qty = qty
        self.price = price
        self.fee = fee
        self.commission = commission
        self.order = order
        
    def __repr__(self):
        return '{} {:%Y-%m-%d %H:%M} qty: {} prc: {}{}{} order: {}'.format(self.symbol, pd.Timestamp(self.date).to_pydatetime(), 
                                                self.qty, self.price, 
                                                ' ' + str(self.fee) if self.fee != 0 else '', 
                                                ' ' + str(self.commission) if self.commission != 0 else '', 
                                                self.order)


class Portfolio:
    '''A portfolio contains one or more strategies that run concurrently so you can test running strategies that are uncorrelated together.'''
    def __init__(self, name = 'main'):
        '''Args:
            name: String used for displaying this portfolio
        '''
        self.name = name
        self.strategies = {}
        
    def add_strategy(self, name, strategy):
        '''
        Args:
            name: Name of the strategy
            strategy: Strategy object
        '''
        self.strategies[name] = strategy
        strategy.portfolio = self
        strategy.name = name
        
    def run_indicators(self, strategy_names = None):
        '''Compute indicators for the strategies specified
        
        Args:
            strategy_names: A list of strategy names.  By default this is set to None and we use all strategies.
        '''
        if strategy_names is None: strategy_names = list(self.strategies.keys())
        if len(strategy_names) == 0: raise Exception('a portofolio must have at least one strategy')
        for name in strategy_names: self.strategies[name].run_indicators()
                
    def run_signals(self, strategy_names = None):
        '''Compute signals for the strategies specified.  Must be called after run_indicators
        
        Args:
            strategy_names: A list of strategy names.  By default this is set to None and we use all strategies.
        '''
        if strategy_names is None: strategy_names = list(self.strategies.keys())
        if len(strategy_names) == 0: raise Exception('a portofolio must have at least one strategy')
        for name in strategy_names: self.strategies[name].run_signals()
                
    def run_rules(self, strategy_names = None, start_date = None, end_date = None, run_first = False, run_last = True):
        '''Run rules for the strategies specified.  Must be called after run_indicators and run_signals.  
          See run function for argument descriptions
        '''
        start_date, end_date = str2date(start_date), str2date(end_date)
        if strategy_names is None: strategy_names = list(self.strategies.keys())
        if len(strategy_names) == 0: raise Exception('a portofolio must have at least one strategy')

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
        '''
        Run indicators, signals and rules.
        
        Args:
            strategy_names: A list of strategy names.  By default this is set to None and we use all strategies.
            start_date: Run rules starting from this date.  
              Sometimes we have a few strategies in a portfolio that need different lead times before they are ready to trade
              so you can set this so they are all ready by this date.  Default None
            end_date: Don't run rules after this date.  Default None
            run_first: Force running rules on the first bar even if signals do not require this.  Default False
            run_last: Force running rules on penultimate bar even if signals do not require this.  
        '''
        start_date, end_date = str2date(start_date), str2date(end_date)
        self.run_indicators()
        self.run_signals()
        return self.run_rules(strategy_names, start_date, end_date, run_first, run_last)
        
    def df_returns(self, sampling_frequency = 'D', strategy_names = None):
        '''
        Return dataframe containing equity and returns with a date index.  Equity and returns are combined from all strategies passed in.
        
        Args:
            sampling_frequency: Date frequency for rows.  Default 'D' for daily so we will have one row per day
            strategy_names: A list of strategy names.  By default this is set to None and we use all strategies.
        '''
        if strategy_names is None: strategy_names = list(self.strategies.keys())
        if len(strategy_names) == 0: raise Exception('portfolio must have at least one strategy')
        equity_list = []
        for name in strategy_names:
            equity = self.strategies[name].df_returns(sampling_frequency = sampling_frequency)[['equity']]
            equity.columns = [name]
            equity_list.append(equity)
        df = pd.concat(equity_list, axis = 1)
        df['equity'] = df.sum(axis = 1)
        df['ret'] = df.equity.pct_change()
        return df
        
    def evaluate_returns(self, sampling_frequency = 'D', strategy_names = None, plot = True, float_precision = 4):
        '''Returns a dictionary of common return metrics.
        
        Args:
            sampling_frequency: Date frequency.  Default 'D' for daily so we downsample to daily returns before computing metrics
            strategy_names: A list of strategy names.  By default this is set to None and we use all strategies.
            plot: If set to True, display plots of equity, drawdowns and returns.  Default False
            float_precision: Number of significant figures to show in returns.  Default 4
        '''
        returns = self.df_returns(sampling_freq, strategy_names)
        ev = compute_return_metrics(returns.index.values, returns.ret.values, returns.equity.values[0])
        display_return_metrics(ev.metrics(), float_precision = float_precision)
        if plot: plot_return_metrics(ev.metrics())
        return ev.metrics()
    
    def plot(self, sampling_frequency = 'D', strategy_names = None):
        '''Display plots of equity, drawdowns and returns
        
        Args:
            sampling_frequency: Date frequency.  Default 'D' for daily so we downsample to daily returns before computing metrics
            strategy_names: A list of strategy names.  By default this is set to None and we use all strategies.
        '''
        returns = self.df_returns(sampling_frequency, strategy_names)
        ev = compute_return_metrics(returns.index.values, returns.ret.values, returns.equity.values[0])
        plot_return_metrics(ev.metrics())
        
    def __repr__(self):
        return f'{self.name} {self.strategies.keys()}'
        
        
class ContractPNL:
    '''Computes pnl for a single contract over time given trades and market data'''
    def __init__(self, contract):
        self.symbol = contract.symbol
        self.multiplier = contract.multiplier
        self.marketdata = contract.marketdata
        self.dates = self.marketdata.dates
        self.unrealized = np.empty(len(self.dates), dtype = np.float) * np.nan; self.unrealized[0] = 0
        self.realized = np.empty(len(self.dates), dtype = np.float) * np.nan; self.realized[0] = 0
        
        #TODO: Add commission and fee from trades
        self.commission = np.zeros(len(self.dates), dtype = np.float);
        self.fee = np.zeros(len(self.dates), dtype = np.float);
        
        self.net_pnl = np.empty(len(self.dates), dtype = np.float) * np.nan; self.net_pnl[0] = 0
        self.position = np.empty(len(self.dates), dtype = np.float) * np.nan; self.position[0] = 0
        self.close = self.marketdata.c
        self._trades = []
        self.open_trades = deque()
        
    def add_trades(self, trades):
        '''Args:
            trades: A list of Trade objects
        '''
        self._trades += trades
        
    def calc(self, prev_i, i):
        '''Compute pnl and store it internally
        
        Args:
            prev_i: Start index to compute pnl from
            i: End index to compute pnl to
        '''
        calc_trades = deque([trade for trade in self._trades if trade.date > self.dates[prev_i] and trade.date <= self.dates[i]])
        
        if not np.isfinite(self.close[i]):
            unrealized = self.unrealized[prev_i]
            realized = 0. 
        else:
            open_trades, unrealized, realized = _calc_pnl(self.open_trades, calc_trades, self.close[i], self.multiplier)
            self.open_trades = open_trades
            
        self.unrealized[i] = unrealized
        self.realized[i] = self.realized[prev_i] + realized
        trade_qty = sum([trade.qty for trade in calc_trades])
        self.position[i] = self.position[prev_i] + trade_qty
        self.net_pnl[i] = self.realized[i] + self.unrealized[i] - self.commission[i] - self.fee[i]
        if np.isnan(self.net_pnl[i]):
            raise Exception(f'net_pnl: nan i: {i} realized: {self.realized[i]} unrealized: {self.unrealized[i]} commission: {self.commission[i]} fee: {self.fee[i]}')
        
    def trades(self, start_date = None, end_date = None):
        '''Get a list of trades
        
        Args:
            start_date: A string or numpy datetime64.  Trades with trade dates >= start_date will be returned.  Default None
            end_date: A string or numpy datetime64.  Trades with trade dates <= end_date will be returned.  Default None
        '''
        start_date, end_date = str2date(start_date), str2date(end_date)
        trades = [trade for trade in self._trades if (start_date is None or trade.date >= start_date) and (end_date is None or trade.date <= end_date)]
        return trades
         
    def df(self):
        '''Returns a pandas dataframe with pnl data, indexed by date'''
        df = pd.DataFrame({'date' : self.dates, 'unrealized' : self.unrealized, 'realized' : self.realized, 
                           'fee' : self.fee, 'net_pnl' : self.net_pnl, 'position' : self.position})
        df.dropna(subset = ['unrealized', 'realized'], inplace = True)
        df['symbol'] = self.symbol
        return df[['symbol', 'date', 'unrealized', 'realized', 'fee', 'net_pnl', 'position']].set_index('date')
    
class Contract:
    '''A Contract can be a real or virtual instrument. For example, for futures you may wish to create a single continous contract instead of
       a contract for each future series
    '''
    def __init__(self, symbol, marketdata, multiplier = 1.):
        '''
        Args:
            symbol: A unique string reprenting this contract. e.g IBM or WTI_FUTURE
            multiplier: If you have to multiply price to get price per contract, set that multiplier there.
            marketdata: A MarketData object containing prices for this contract.
        '''
        assert(isinstance(symbol, str) and len(symbol) > 0)
        assert(multiplier > 0)
        self.symbol = symbol
        self.multiplier = multiplier
        self.marketdata = marketdata

class Account:
    '''An Account calculates pnl for a set of contracts'''
    def __init__(self, contracts, starting_equity = 1.0e6, calc_frequency = 'D'):
        '''
        Args:
            contracts: A list of Contract objects
            starting_equity: Starting equity in account currency.  Default 1.e6
            calc_frequency: Account will calculate pnl at this frequency.  Default 'D' for daily
        '''
        if calc_frequency != 'D': raise Exception('unknown calc frequency: {}'.format(calc_frequency))
        self.calc_freq = calc_frequency
        self.contract_pnls = defaultdict()
        self.current_calc_index = 0
        self.marketdata = {}
        self.all_dates = None
        self.starting_equity = starting_equity
        if len(contracts) == 0:
            raise Exception('must add at least one contract')
        for contract in contracts: 
            self.add_contract(contract)
    
    def _set_dates(self, dates):
        if self.all_dates is not None and not np.array_equal(dates, all_dates):
            raise Exception('all symbols in a strategy must have the same dates')
        self.all_dates = dates
        calc_dates = dates.astype('M8[D]')
        self.calc_dates = np.unique(calc_dates)
        self.calc_indices = np.searchsorted(dates, self.calc_dates, side='left') - 1
        if self.calc_indices[0] == -1: self.calc_indices[0] = 0
        self._equity = np.empty(len(dates), np.float) * np.nan; 
        self._equity[0] = self.starting_equity
        
    def symbols(self):
        return list(self.contract_pnls.keys())
        
    def add_contract(self, contract):
        if self.all_dates is None: self._set_dates(contract.marketdata.dates)
        self.contract_pnls[contract.symbol] = ContractPNL(contract)
        self.marketdata[contract.symbol] = contract.marketdata
        
    def _add_trades(self, symbol, trades):
        self.contract_pnls[symbol].add_trades(trades)
        
    def calc(self, i):
        '''
        Computes P&L and stores it internally for all contracts.
        
        Args:
            i: Index to compute P&L at.  Account remembers the last index it computed P&L up to and will compute P&L between these two indices
        '''
        calc_indices = self.calc_indices[:]
        if self.current_calc_index == i: return
        intermediate_calc_indices = np.ravel(np.where(np.logical_and(calc_indices > self.current_calc_index, calc_indices <= i)))

        if not len(intermediate_calc_indices) or calc_indices[intermediate_calc_indices[-1]] != i: 
            calc_indices = np.append(calc_indices, i)
            intermediate_calc_indices = np.append(intermediate_calc_indices, len(calc_indices) - 1)
            
        for symbol, symbol_pnl in self.contract_pnls.items():
            prev_calc_index = self.current_calc_index
            for idx in intermediate_calc_indices:
                calc_index = calc_indices[idx]
                symbol_pnl.calc(prev_calc_index, calc_index)
                self._equity[calc_index] = self._equity[prev_calc_index] + symbol_pnl.net_pnl[calc_index] - symbol_pnl.net_pnl[prev_calc_index]
                # print(f'prev_calc_index: {prev_calc_index} calc_index: {calc_index} prev_equity: {self._equity[prev_calc_index]} net_pnl: {symbol_pnl.net_pnl[calc_index]} prev_net_pnl: {symbol_pnl.net_pnl[prev_calc_index]}')
                prev_calc_index = calc_index
                
        self.current_calc_index = i
        
    def position(self, symbol, date):
        '''Returns position for a symbol at a given date in number of contracts or shares.  Will cause calculation if Account has not previously calculated
          up to this date'''
        i = self.find_index_before(date)
        self.calc(i)
        return self.contract_pnls[symbol].position[i]
    
    def equity(self, date):
        '''Returns equity in this account in Account currency.  Will cause calculation if Account has not previously calculated up to this date'''
        i = self.find_index_before(date)
        self.calc(i)
        return self._equity[i]
    
    def trades(self, symbol = None, start_date = None, end_date = None):
        '''Returns a list of trades with the given symbol and with trade date between (and including) start date and end date if they are specified.
          If symbol is None trades for all symbols are returned'''
        start_date, end_date = str2date(start_date), str2date(end_date)
        if symbol is None:
            trades = []
            for symbol, sym_pnl in self.contract_pnls.items():
                trades += sym_pnl.trades(start_date, end_date)
            return trades
        else:
            return self.contract_pnls[symbol].trades(start_date, end_date)
        
    def find_index_before(self, date):
        '''Returns the market data index before or at date'''
        return np.searchsorted(self.all_dates, date)
        
    def transfer_cash(self, date, amount):
        '''Move cash from one portfolio to another'''
        i = self.find_index_before(date)
        curr_equity = self.equity(date)
        if (amount > curr_equity): amount = curr_equity # Cannot make equity negative
        self._equity[i] -= amount
        return amount

    def df_pnl(self, symbol = None):
        '''Returns a dataframe with P&L columns.  If symbol is set to None (default), sums up P&L across symbols'''
        if symbol:
            ret = self.contract_pnls[symbol].df()
        else:
            dfs = []
            for symbol, symbol_pnl in self.contract_pnls.items():
                df = symbol_pnl.df()
                dfs.append(df)
            ret = pd.concat(dfs)
            ret = ret.reset_index().groupby('date').sum()
        df_equity = pd.DataFrame({'equity' : self._equity}, index = self.all_dates).dropna()
        ret = pd.merge(ret, df_equity, left_index = True, right_index = True, how = 'outer')
        ret.index.name = 'date'
        return ret
    
    def df_trades(self, symbol = None, start_date = None, end_date = None):
        '''Returns a dataframe with data from trades with the given symbol and with trade date between (and including) start date and end date
          if they are specified.  If symbol is None, trades for all symbols are returned'''
        start_date, end_date = str2date(start_date), str2date(end_date)
        if symbol:
            trades = self.contract_pnls[symbol].trades(start_date, end_date)
        else:
            trades = [v.trades(start_date, end_date) for v in self.contract_pnls.values()]
            trades = [trade for sublist in trades for trade in sublist] # flatten list
        df = pd.DataFrame.from_records([(trade.symbol, trade.date, trade.qty, trade.price, trade.fee, trade.commission, trade.order.date, trade.order.qty, trade.order.params()) for trade in trades],
                    columns = ['symbol', 'date', 'qty', 'price', 'fee', 'commission', 'order_date', 'order_qty', 'order_params'])
        return df
    
class Strategy:
    def __init__(self, contracts, starting_equity = 1.0e6, calc_frequency = 'D'):
        '''
        Args:
            contracts: A list of contract objects
            starting_equity: Starting equity in Strategy currency.  Default 1.e6
            calc_frequency: How often P&L is calculated.  Default is 'D' for daily
        '''
        self.name = None
        self.account = Account(contracts, starting_equity, calc_frequency)
        self.symbols = [contract.symbol for contract in contracts]
        self.indicators = {}
        self.indicator_values = defaultdict(dict)
        self.signals = {}
        self.signal_values = defaultdict(dict)
        self.rules = {}
        self.rule_signals = {}
        self.market_sims = {}
        self._trades = defaultdict(list)
        self._orders = []
        self.dates = self.account.all_dates
        
    def add_indicator(self, name, indicator_function):
        '''
        Args:
            name: Name of the indicator
            indicator_function:  A function taking a MarketData object and returning a numpy array
              containing indicator values.  The return array must have the same length as the MarketData object
        '''
        self.indicators[name] = indicator_function
        
    def add_signal(self, name, signal_function):
        '''
        Args:
            name: Name of the signal
            signal_function:  A function taking a MarketData object and a dictionary of indicator value arrays as input and returning a numpy array
              containing signal values.  The return array must have the same length as the MarketData object
        '''
        self.signals[name] = signal_function
        
    def add_rule(self, name, rule_function, signal_name, sig_true_values):
        '''Add a trading rule
        
        Args:
            name: Name of the trading rule
            rule_function: A trading rule function that returns a list of Orders
            signal_name: The strategy will call the trading rule function when the signal with this name matches sig_true_values
            sig_true_values: A numpy array of values.  If the signal value at a bar is equal to one of these, the Strategy will call the trading rule function
        '''
        self.rule_signals[name] = (signal_name, sig_true_values)
        self.rules[name] = rule_function
        
    def add_market_sim(self, market_sim_function, symbols = None):
        '''Add a market simulator.  A market simulator takes a list of Orders as input and returns a list of Trade objects.
        
        Args:
            market_sim_function: A function that takes a list of Orders and MarketData as input and returns a list of Trade objects
            symbols: A list of the symbols that this market_sim_function applies to. If None (default) it will apply to all symbols
        '''
        if symbols is None: symbols = self.symbols
        for symbol in symbols: self.market_sims[symbol] = market_sim_function
        
    def run_indicators(self, indicator_names = None, symbols = None):
        '''Calculate values of the indicators specified and store them.
        
        Args:
            indicator_names: List of indicator names.  If None (default) run all indicators
            symbols: List of symbols to run these indicators for.  If None (default) use all symbols
        '''
        if indicator_names is None: indicator_names = self.indicators.keys()
        if symbols is None: symbols = self.symbols
            
        for indicator_name in indicator_names:
            indicator_function = self.indicators[indicator_name]
            for symbol in symbols:
                marketdata = self.account.marketdata[symbol]
                self.indicator_values[symbol][indicator_name] = series_to_array(indicator_function(marketdata))
                
    def run_signals(self, signal_names = None, symbols = None):
        '''Calculate values of the signals specified and store them.
        
        Args:
            signal_names: List of signal names.  If None (default) run all signals
            symbols: List of symbols to run these signals for.  If None (default) use all symbols
        '''
        if signal_names is None: signal_names = self.signals.keys()
        if symbols is None: symbols = self.symbols
        
        for signal_name in signal_names:
            signal_function = self.signals[signal_name]
            for symbol in symbols:
                marketdata = self.account.marketdata[symbol]
                self.signal_values[symbol][signal_name] = series_to_array(signal_function(marketdata, self.indicator_values[symbol]))
                
    def run_rules(self, rule_names = None, symbols = None, start_date = None, end_date = None, run_first = False, run_last = True):
        '''Run trading rules.
        
        Args:
            rule_names: List of rule names.  If None (default) run all rules
            symbols: List of symbols to run these signals for.  If None (default) use all symbols
            start_date: Run rules starting from this date. Default None 
            end_date: Don't run rules after this date.  Default None
            run_first: Force running rules on the first bar even if signals do not require this.  Default False
            run_last: Force running rules on penultimate bar even if signals do not require this.  
        '''
        start_date, end_date = str2date(start_date), str2date(end_date)
        dates, iterations = self._get_iteration_indices(rule_names, symbols, start_date, end_date, run_first, run_last)
        # Now we know which rules, symbols need to be applied for each iteration, go through each iteration and apply them
        # in the same order they were added to the strategy
        for i, tup_list in enumerate(iterations):
            self._iterate(i, tup_list)
        
    def _get_iteration_indices(self, rule_names = None, symbols = None, start_date = None, end_date = None, 
                  run_first = False, run_last = True):
        start_date, end_date = str2date(start_date), str2date(end_date)
        if rule_names is None: rule_names = self.rules.keys()
        if symbols is None: symbols = self.symbols
            
        dates = self.dates
        num_dates = len(dates)
                    
        iterations = [[] for x in range(num_dates)]
        self.orders_iter = [[] for x in range(num_dates)]
            
        for rule_name in rule_names:
            rule_function = self.rules[rule_name]
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
                for idx in indices: iterations[idx].append((rule_function, symbol, iteration_params))
                    
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
                rule_function, symbol, params = tup
                open_orders = self._get_orders(i, rule_function, symbol, params)
                self._orders += open_orders
                if len(open_orders): self.orders_iter[i + 1].append((open_orders, symbol, params))
            except Exception as e:
                raise type(e)(f'Exception: {str(e)} at rule: {type(tup[0])} symbol: {tup[1]} index: {i}').with_traceback(sys.exc_info()[2])
                    
    def _get_orders(self, idx, rule_function, symbol, params):
        indicator_values, signal_values, marketdata = (params['indicator_values'], params['signal_values'], params['marketdata'])
        open_orders = rule_function(self, symbol, idx, self.dates[idx], marketdata, indicator_values, signal_values, self.account)
        return open_orders
        
    def _sim_market(self, idx, open_orders, symbol, params):
        '''
        Keep iterating while we have open orders since they may get filled
        TODO: For limit orders and trigger orders we can be smarter here and reduce indices like quantstrat does
        '''
        market_sim_function = params['market_sim']
        trades = market_sim_function(self, open_orders, idx, self.dates[idx], self.account.marketdata[symbol])
        if len(trades) == 0: return []
        self._trades[symbol] += trades
        self.account._add_trades(symbol, trades)
        self.account.calc(idx)
        open_orders = [order for order in open_orders if order.status == 'open']
        return open_orders
            
    def df_data(self, symbols = None, add_pnl = True, start_date = None, end_date = None):
        '''
        Add indicators and signals to end of market data and return as a pandas dataframe.
        
        Args:
            symbols: list of symbols to include.  All if set to None (default)
            add_pnl: If True (default), include P&L columns in dataframe
            start_date: string or numpy datetime64. Default None
            end_date: string or numpy datetime64: Default None
        '''
        start_date, end_date = str2date(start_date), str2date(end_date)
        if symbols is None: symbols = self.symbols
        if not isinstance(symbols, list): symbols = [symbols]
            
        mds = []
            
        for symbol in symbols:
            md = self.account.marketdata[symbol].df(start_date, end_date)
            
            md.insert(0, 'symbol', symbol)
            if add_pnl: 
                df_pnl = self.account.df_pnl(symbol)
                del df_pnl['symbol']

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
            
            if add_pnl: md = pd.merge(md, df_pnl, left_index = True, right_index = True, how = 'left')
            # Add counter column for debugging
            md.insert(len(md.columns), 'i', np.arange(len(md)))
            
            mds.append(md)
            
        return pd.concat(mds)
    
    def marketdata(self, symbol):
        '''Return MarketData object for this symbol'''
        return self.account.marketdata[symbol]
    
    def trades(self, symbol = None, start_date = None, end_date = None):
        '''Returns a list of trades with the given symbol and with trade date between (and including) start date and end date if they are specified.
          If symbol is None trades for all symbols are returned'''
        start_date, end_date = str2date(start_date), str2date(end_date)
        return self.account.trades(symbol, start_date, end_date)
    
    def df_trades(self, symbol = None, start_date = None, end_date = None):
        '''Returns a dataframe with data from trades with the given symbol and with trade date between (and including) start date and end date
          if they are specified.  If symbol is None, trades for all symbols are returned'''
        start_date, end_date = str2date(start_date), str2date(end_date)
        return self.account.df_trades(symbol, start_date, end_date)
    
    def orders(self, symbol = None, start_date = None, end_date = None):
        '''Returns a list of orders with the given symbol and with order date between (and including) start date and end date if they are specified.
          If symbol is None orders for all symbols are returned'''
        start_date, end_date = str2date(start_date), str2date(end_date)
        return [order for order in self._orders if (symbol is None or order.symbol == symbol) and (
            start_date is None or order.date >= start_date) and (end_date is None or order.date <= end_date)]
    
    def df_orders(self, symbol = None, start_date = None, end_date = None):
        '''Returns a dataframe with data from orders with the given symbol and with order date between (and including) start date and end date
          if they are specified.  If symbol is None, orders for all symbols are returned'''
        start_date, end_date = str2date(start_date), str2date(end_date)
        orders = self.orders(symbol, start_date, end_date)
        df_orders = pd.DataFrame.from_records([(order.symbol, type(order).__name__, order.date, order.qty, order.params()) 
                                               for order in orders], columns = ['symbol', 'type', 'date', 'qty', 'params'])
        return df_orders
   
    def df_pnl(self, symbol = None):
        '''Returns a dataframe with P&L columns.  If symbol is set to None (default), sums up P&L across symbols'''
        return self.account.df_pnl(symbol)
    
    def df_returns(self, symbol = None, sampling_frequency = 'D'):
        '''Return a dataframe of returns and equity indexed by date.
        
        Args:
            symbol: The symbol to get returns for.  If set to None (default), this returns the sum of PNL for all symbols
            sampling_frequency: Downsampling frequency.  Default is None.  See pandas frequency strings for possible values
        '''
        pnl = self.df_pnl(symbol)[['equity']]
        pnl.equity = pnl.equity.ffill()
        pnl = pnl.resample(sampling_frequency).last()
        pnl['ret'] = pnl.equity.pct_change()
        return pnl
    
    def plot(self, symbols = None, md_columns = 'c', pnl_columns = 'equity', title = None, figsize = (20, 15), date_range = None, 
             date_format = None, sampling_frequency = None, trade_marker_properties = None, hspace = 0.15):
        
        '''Plot indicators, signals, trades, position, pnl
        
        Args:
            symbols: List of symbols or None (default) for all symbols
            md_columns: List of columns of market data to plot.  Default is 'c' for close price.  You can set this to 'ohlcv' if you want to plot
             a candlestick of OHLCV data
            pnl_columns: List of P&L columns to plot.  Default is 'equity'
            title: Title of plot (None)
            figsize: Figure size.  Default is (20, 15)
            date_range: Tuple of strings or datetime64, e.g. ("2018-01-01", "2018-04-18 15:00") to restrict the graph.  Default None
            date_format: Date format for tick labels on x axis.  If set to None (default), will be selected based on date range. See matplotlib date format strings
            sampling_frequency: Downsampling frequency.  Default is None.  The graph may get too busy if you have too many bars of data, in which case you may want to 
                downsample before plotting.  See pandas frequency strings for possible values
            trade_marker_properties: A dictionary of order reason code -> marker shape, marker size, marker color for plotting trades with different reason codes.
              Default is None in which case the dictionary from the ReasonCode class is used
            hspace: Height (vertical) space between subplots.  Default is 0.15
        '''
        date_range = strtup2date(date_range)
        if symbols is None: symbols = self.symbols
        if not isinstance(symbols, list): symbols = [symbols]
        if not isinstance(md_columns, list): md_columns = [md_columns]
        if not isinstance(pnl_columns, list): pnl_columns = [pnl_columns]
        for symbol in symbols:
            md = self.marketdata(symbol)
            md_dates = md.dates
            if md_columns == ['ohlcv']:
                md_list = [OHLC('price', dates = md_dates, o = md.o, h = md.h, l = md.l, c = md.c, v = md.v)]
            else:
                md_list = [TimeSeries(md_column, dates = md_dates, values = getattr(md, md_column)) for md_column in md_columns]
            indicator_list = [TimeSeries(indicator_name, dates = md_dates, values = self.indicator_values[symbol][indicator_name], line_type = '--'
                                        ) for indicator_name in self.indicators.keys() if indicator_name in self.indicator_values[symbol]]
            signal_list = [TimeSeries(signal_name, dates = md_dates, values = self.signal_values[symbol][signal_name]
                                     ) for signal_name in self.signals.keys() if signal_name in self.signal_values[symbol]]
            df_pnl_ = self.df_pnl(symbol)
            pnl_list = [TimeSeries(pnl_column, dates = df_pnl_.index.values, values = df_pnl_[pnl_column].values) for pnl_column in pnl_columns]
            if trade_marker_properties:
                trade_sets = trade_sets_by_reason_code(self._trades[symbol], trade_marker_properties)
            else:
                trade_sets = trade_sets_by_reason_code(self._trades[symbol])
            main_subplot = Subplot(indicator_list +  md_list + trade_sets, height_ratio = 0.5, ylabel = 'Indicators')
            signal_subplot = Subplot(signal_list, ylabel = 'Signals', height_ratio = 0.167)
            pnl_subplot = Subplot(pnl_list, ylabel = 'Equity', height_ratio = 0.167, log_y = True, y_tick_format = '${x:,.0f}')
            position = df_pnl_.position.values
            pos_subplot = Subplot([TimeSeries('position', dates = df_pnl_.index.values, values = position, plot_type = 'filled_line')], 
                                  ylabel = 'Position', height_ratio = 0.167)
            plot = Plot([main_subplot, signal_subplot, pos_subplot, pnl_subplot], figsize = figsize,
                                date_range = date_range, date_format = date_format, sampling_frequency = sampling_frequency, title = title, hspace = hspace)
            plot.draw()
            
    def evaluate_returns(self, symbol = None, plot = True, float_precision = 4):
        '''Returns a dictionary of common return metrics.
        
        Args:
            sampling_frequency: Date frequency.  Default 'D' for daily so we downsample to daily returns before computing metrics
            strategy_names: A list of strategy names.  By default this is set to None and we use all strategies.
            plot: If set to True, display plots of equity, drawdowns and returns.  Default False
            float_precision: Number of significant figures to show in returns.  Default 4
        '''
        returns = self.df_returns(symbol)
        ev = compute_return_metrics(returns.index.values, returns.ret.values, self.account.starting_equity)
        display_return_metrics(ev.metrics(), float_precision = float_precision)
        if plot: plot_return_metrics(ev.metrics())
        return ev.metrics()
    
    def plot_returns(self, symbol = None):
        '''Display plots of equity, drawdowns and returns for the given symbol or for all symbols if symbol is None (default)'''
        if symbol is None:
            symbols = self.symbols()
        else:
            symbols = [symbol]
            
        df_list = []
            
        for symbol in symbols:
            df_list.append(self.df_returns(symbol))
        
        df = pd.concat(df_list, axis = 1)
            
        ev = compute_return_metrics(returns.index.values, returns.ret.values, self.account.starting_equity)
        plot_return_metrics(ev.metrics())
       
    def __repr__(self):
        return f'{pformat(self.indicators)} {pformat(self.rules)} {pformat(self.account)}'

def test_strategy(): 
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
        reason_code = None
        
        if not marketdata.valid_row(i): return []
        
        curr_pos = account.position(symbol, date)
        
        if i == len(marketdata.dates) - 2: # Last date so get out of position
            if not math.isclose(curr_pos, 0): 
                return [MarketOrder(symbol, date, -curr_pos, reason_code = ReasonCode.BACKTEST_END)]
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
            reason_code = ReasonCode.ENTER_LONG if order_qty > 0 else ReasonCode.ENTER_SHORT
        else:
            order_qty = -curr_pos
            trigger_price = stop_limit
            reason_code = ReasonCode.EXIT_LONG if order_qty < 0 else ReasonCode.EXIT_SHORT
        
        order_qty = round(order_qty)
        
        if np.isnan(order_qty):
            raise Exception(f'Got nan order qty date: {date} i: {i} curr_pos: {curr_pos} curr_equity: {curr_equity} entry_limit: {entry_limit} stop_limit: {stop_limit}')
            
        if math.isclose(order_qty, 0): return []
        
        order = StopLimitOrder(symbol, date, order_qty, trigger_price, reason_code = reason_code)
            
        return [order]
    
    def get_support(lows, n): return pd.Series(lows).rolling(window = n, min_periods = 1).min().values

    def get_resistance(highs, n): return pd.Series(highs).rolling(window = n, min_periods = 1).max().values
    
    def get_trend(md, ind):
        trend = pd.Series(np.where(pd.Series(md.h) > shift_np(ind['slow_resistance'], 1), 1, 
                          np.where(pd.Series(md.l) < shift_np(ind['slow_support'], 1), -1, 
                          np.nan)))
        trend.fillna(method = 'ffill', inplace = True)
        return trend.values
    
    def build_strategy(contract, fast_interval, slow_interval):
        strategy = Strategy([contract])
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
        contract = Contract('IBM', MarketData(dates, c, o, h, l, v))
        portfolio.add_strategy(f'strat_{days}', build_strategy(contract, fast_interval, slow_interval))
    
    # Start at max slow days so all strategies start at the same time
    print('running')
    portfolio.run(start_date = dates[slow_interval])
    print('done')
    
    strat1 = portfolio.strategies['strat_0.5']
    portfolio.plot();
    strat1.plot();
    
if __name__ == "__main__":
    test_strategy()

