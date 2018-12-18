#cell 0
import numpy as np
import pandas as pd
import sys
from collections import defaultdict
from pprint import pformat
import math

from pyqstrat.pq_utils import *
from pyqstrat.marketdata import *
from pyqstrat.evaluator import compute_return_metrics, display_return_metrics, plot_return_metrics
from pyqstrat.account import Account

#cell 1
class Strategy:
    def __init__(self, contracts, marketdata_collection, starting_equity = 1.0e6, calc_frequency = 'D', additional_order_dates = None, additional_trade_dates = None):
        '''
        Args:
            contracts (list of Contract): The contracts we will potentially trade
            starting_equity (float, optional): Starting equity in Strategy currency.  Default 1.e6
            calc_frequency (str, optional): How often P&L is calculated.  Default is 'D' for daily
            additional_account_dates (np.array of np.datetime64, optional): If present, we check for orders on these dates.  Default None
            additional_tradedates (np.array of np.datetime64, optional): If present, we check for trades on these dates.  Default None
        '''
        self.name = None
        date_list = []
        if additional_order_dates is not None: date_list.append(additional_order_dates)
        if additional_trade_dates is not None: date_list.append(additional_trade_dates)
        self.additional_order_dates = additional_order_dates
        self.additional_trade_dates = additional_trade_dates
        if len(date_list): marketdata_collection.add_dates(np.concatenate(date_list))
        self.dates = marketdata_collection.dates()
        self.account = Account(contracts, marketdata_collection, starting_equity, calc_frequency)
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
        
    def add_rule(self, name, rule_function, signal_name, sig_true_values = None):
        '''Add a trading rule
        
        Args:
            name (str): Name of the trading rule
            rule_function (function): A trading rule function that returns a list of Orders
            signal_name (str): The strategy will call the trading rule function when the signal with this name matches sig_true_values
            sig_true_values (numpy array, optional): If the signal value at a bar is equal to one of these values, the Strategy will call the trading rule function.  
                Default [TRUE]
        '''
        if sig_true_values is None: sig_true_values = [True]
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
                
    def run_rules(self, rule_names = None, symbols = None, start_date = None, end_date = None):
        '''Run trading rules.
        
        Args:
            rule_names: List of rule names.  If None (default) run all rules
            symbols: List of symbols to run these signals for.  If None (default) use all symbols
            start_date: Run rules starting from this date. Default None 
            end_date: Don't run rules after this date.  Default None
        '''
        start_date, end_date = str2date(start_date), str2date(end_date)
        dates, orders_iter, trades_iter = self._get_iteration_indices(rule_names, symbols, start_date, end_date)
        # Now we know which rules, symbols need to be applied for each iteration, go through each iteration and apply them
        # in the same order they were added to the strategy
        for i, tup_list in enumerate(orders_iter):
            self._check_for_trades(i, trades_iter[i])
            self._check_for_orders(i, tup_list)
        
    def _get_iteration_indices(self, rule_names = None, symbols = None, start_date = None, end_date = None):
        '''
        >>> class MockStrat:
        ...    def __init__(self):
        ...        self.dates = dates
        ...        self.account = self
        ...        self.additional_order_dates = None
        ...        self.additional_trade_dates = np.array(['2018-01-03'], dtype = 'M8[D]')
        ...        self.rules = {'rule_a' : rule_a, 'rule_b' : rule_b}
        ...        self.marketdata = {'IBM' : self, 'AAPL' : self}
        ...        self.market_sims = {'IBM' : market_sim_ibm, 'AAPL' : market_sim_aapl}
        ...        self.rule_signals = {'rule_a' : ('sig_a', [1]), 'rule_b' : ('sig_b', [1, -1])}
        ...        self.signal_values = {'IBM' : {'sig_a' : np.array([0., 1., 1.]), 'sig_b' : np.array([0., 0., 0.])},
        ...                              'AAPL' : {'sig_a' : np.array([0., 0., 0.]), 'sig_b' : np.array([0., -1., -1])}}
        ...        self.indicator_values = {'IBM' : None, 'AAPL' : None}
        >>>
        >>> def market_sim_aapl(): pass
        >>> def market_sim_ibm(): pass
        >>> def rule_a(): pass
        >>> def rule_b(): pass
        >>> dates = np.array(['2018-01-01', '2018-01-02', '2018-01-03'], dtype = 'M8[D]')
        >>> rule_names = ['rule_a', 'rule_b']
        >>> symbols = ['IBM', 'AAPL']
        >>> start_date = np.datetime64('2018-01-01')
        >>> end_date = np.datetime64('2018-02-05')
        >>> dates, orders_iter, trades_iter = Strategy._get_iteration_indices(MockStrat(), rule_names, symbols, start_date, end_date)
        >>> assert(len(trades_iter[1]) == 0)
        >>> assert(trades_iter[2][1][1] == "AAPL")
        >>> assert(trades_iter[2][2][1] == "IBM")
        >>> assert(len(orders_iter[0]) == 0)
        >>> assert(len(orders_iter[1]) == 2)
        >>> assert(orders_iter[1][0][1] == "IBM")
        >>> assert(orders_iter[1][1][1] == "AAPL")
        >>> assert(len(orders_iter[2]) == 0)
        '''
        start_date, end_date = str2date(start_date), str2date(end_date)
        if rule_names is None: rule_names = self.rules.keys()
        if symbols is None: symbols = self.symbols

        num_dates = len(self.dates)

        orders_iter = [[] for x in range(num_dates)]
        trades_iter = [[] for x in range(num_dates)]

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

                if self.additional_order_dates is not None:
                    additional_indices = np.searchsorted(self.dates, self.additional_order_dates)
                    indices = np.sort(np.unique(np.concatenate([indices, additional_indices])))

                if len(indices) and indices[-1] == len(sig_values) -1: indices = indices[:-1] # Don't run rules on last index since we cannot fill any orders

                indicator_values = self.indicator_values[symbol]
                iteration_params = {'market_sim' : market_sim, 'indicator_values' : indicator_values, 'signal_values' : sig_values, 'marketdata' : marketdata}
                for idx in indices: orders_iter[idx].append((rule_function, symbol, iteration_params))

                if self.additional_trade_dates is not None:
                    trade_indices = np.sort(np.unique(np.searchsorted(self.dates, self.additional_trade_dates)))
                    for idx in trade_indices: trades_iter[idx].append(([], symbol, iteration_params))

            self.orders_iter = orders_iter
            self.trades_iter = trades_iter # For debugging

        return self.dates, orders_iter, trades_iter
         
    def _check_for_trades(self, i, tup_list):
        for tup in tup_list:
            try:
                open_orders, symbol, params = tup
                open_orders = self._sim_market(i, open_orders, symbol, params)
                if len(open_orders): self.trades_iter[i + 1].append((open_orders, symbol, params))
            except Exception as e:
                raise type(e)(f'Exception: {str(e)} at rule: {type(tup[0])} symbol: {tup[1]} index: {i}').with_traceback(sys.exc_info()[2])
                
    def _check_for_orders(self, i, tup_list):
        for tup in tup_list:
            try:
                rule_function, symbol, params = tup
                open_orders = self._get_orders(i, rule_function, symbol, params)
                self._orders += open_orders
                if len(open_orders): self.trades_iter[i + 1].append((open_orders, symbol, params))
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
                md_list = [OHLC('price', dates = md_dates, o = md.o, h = md.h, l = md.l, c = md.c, v = md.v, vwap = md.vwap)]
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
            symbol (str): Date frequency.  Default 'D' for daily so we downsample to daily returns before computing metrics
            plot (bool): If set to True, display plots of equity, drawdowns and returns.  Default False
            float_precision (float, optional): Number of significant figures to show in returns.  Default 4
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

