#cell 0
import pandas as pd
import numpy as np
from functools import reduce
import datetime

from pyqstrat.pq_utils import *
from pyqstrat.evaluator import compute_return_metrics, display_return_metrics, plot_return_metrics
from pyqstrat.strategy import Strategy

#cell 1
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
        if len(strategy_names) == 0: raise Exception('a portfolio must have at least one strategy')
        for name in strategy_names: self.strategies[name].run_indicators()
                
    def run_signals(self, strategy_names = None):
        '''Compute signals for the strategies specified.  Must be called after run_indicators
        
        Args:
            strategy_names: A list of strategy names.  By default this is set to None and we use all strategies.
        '''
        if strategy_names is None: strategy_names = list(self.strategies.keys())
        if len(strategy_names) == 0: raise Exception('a portfolio must have at least one strategy')
        for name in strategy_names: self.strategies[name].run_signals()
            
    def _get_iterations(self, strategies, start_date, end_date):
        '''
        >>> class Strategy:
        ...    def __init__(self, num): 
        ...        self.num = num
        ...        self.timestamps = [
        ...            np.array(['2018-01-01', '2018-01-02', '2018-01-03'], dtype = 'M8[D]'),
        ...            np.array(['2018-01-02', '2018-01-03', '2018-01-04'], dtype = 'M8[D]')]
        ...    def _check_for_orders(self, args): pass
        ...    def _check_for_trades(self, args): pass
        ...    def _get_iteration_indices(self, start_date, end_date):
        ...        i = self.num
        ...        return self.timestamps[self.num - 1], [f'oarg_1_{1}', f'oarg_2_{i}', f'oarg_3_{i}'], [f'targ_1_{i}', f'targ_2_{i}', f'targ_3_{i}']
        ...    def __repr__(self):
        ...        return f'{self.num}'

        >>> orders_iter, trades_iter = Portfolio._get_iterations(None, [Strategy(1), Strategy(2)], None, None)
        >>> assert(len(orders_iter) == 4)
        >>> assert(len(trades_iter) == 4)
        >>> print(orders_iter[2]) #doctest: +ELLIPSIS
        [(<function Strategy._check_for_orders at ...>, (1, 2, 'oarg_3_1')), (<function Strategy._check_for_orders at ...>, (2, 1, 'oarg_2_2'))]
        >>> print(trades_iter[3]) #doctest: +ELLIPSIS
        [(<function Strategy._check_for_trades at ...>, (2, 2, 'targ_3_2'))]
        '''
        orders_iter_list = []
        trades_iter_list = []

        for strategy in strategies:
            timestamps, orders_iter, trades_iter = strategy._get_iteration_indices(start_date = start_date, end_date = end_date)
            orders_iter_list.append((strategy, timestamps, orders_iter))
            trades_iter_list.append((strategy, timestamps, trades_iter))

        timestamps_list = [tup[1] for tup in orders_iter_list] + [tup[1] for tup in trades_iter_list]
        all_timestamps = np.array(reduce(np.union1d, timestamps_list))
        
        trade_iterations = [[] for x in range(len(all_timestamps))]

        for tup in trades_iter_list: # per strategy
            strategy = tup[0]
            timestamps = tup[1]
            trades_iter = tup[2] # vector with list of (rule, symbol, iter_params dict)

            for i, timestamp in enumerate(timestamps):
                idx = np.searchsorted(all_timestamps, timestamp)
                args = (strategy, i, trades_iter[i])
                trade_iterations[idx].append((Strategy._check_for_trades, args))

        order_iterations = [[] for x in range(len(all_timestamps))]

        for tup in orders_iter_list: # per strategy
            strategy = tup[0]
            timestamps = tup[1]
            orders_iter = tup[2] # vector with list of (rule, symbol, iter_params dict)

            for i, timestamp in enumerate(timestamps):
                idx = np.searchsorted(all_timestamps, timestamp)
                args = (strategy, i, orders_iter[i])
                order_iterations[idx].append((Strategy._check_for_orders, args))

        return order_iterations, trade_iterations
                
    def run_rules(self, strategy_names = None, start_date = None, end_date = None):
        '''Run rules for the strategies specified.  Must be called after run_indicators and run_signals.  
          See run function for argument descriptions
        '''
        start_date, end_date = str2date(start_date), str2date(end_date)
        if strategy_names is None: strategy_names = list(self.strategies.keys())
        if len(strategy_names) == 0: raise Exception('a portfolio must have at least one strategy')

        strategies = [self.strategies[key] for key in strategy_names]
        
        min_date = min([strategy.timestamps[0] for strategy in strategies])
        if start_date: min_date = max(min_date, start_date)
        max_date = max([strategy.timestamps[-1] for strategy in strategies])
        if end_date: max_date = min(max_date, end_date)
            
        order_iterations, trade_iterations = self._get_iterations(strategies, start_date, end_date)
        
        rerun_trades = []
                
        for i, orders_iter in enumerate(order_iterations): # Per timestamp
            trades_iter = trade_iterations[i] 
            for tup in trades_iter: # Per strategy
                func = tup[0]
                args = tup[1]
                strategy = args[0]
                if strategy.trade_lag == 0: # When trade lag is 0, we have to rerun trades after we run orders
                    rerun_trades.append((func, args))
                    continue
                func(*args)
            
            for tup in orders_iter: # Per strategy
                func = tup[0]
                args = tup[1]
                func(*args)
                
            for (func, args) in rerun_trades:
                func(*args)
                
        # Make sure we calc to the end for each strategy
        for strategy in strategies:
            strategy.account.calc(strategy.timestamps[-1])
                
    def run(self, strategy_names = None, start_date = None, end_date = None):
        '''
        Run indicators, signals and rules.
        
        Args:
            strategy_names: A list of strategy names.  By default this is set to None and we use all strategies.
            start_date: Run rules starting from this date.  
              Sometimes we have a few strategies in a portfolio that need different lead times before they are ready to trade
              so you can set this so they are all ready by this date.  Default None
            end_date: Don't run rules after this date.  Default None
         '''
        start_date, end_date = str2date(start_date), str2date(end_date)
        self.run_indicators()
        self.run_signals()
        return self.run_rules(strategy_names, start_date, end_date)
        
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
            equity = self.strategies[name].df_returns(sampling_frequency = sampling_frequency)[['timestamp', 'equity']]
            equity.columns = ['timestamp', name]
            equity = equity.set_index('timestamp')
            equity_list.append(equity)
        df = pd.concat(equity_list, axis = 1)
        df['equity'] = df.sum(axis = 1)
        df['ret'] = df.equity.pct_change()
        return df.reset_index()
        
    def evaluate_returns(self, sampling_frequency = 'D', strategy_names = None, plot = True, float_precision = 4):
        '''Returns a dictionary of common return metrics.
        
        Args:
            sampling_frequency: Date frequency.  Default 'D' for daily so we downsample to daily returns before computing metrics
            strategy_names: A list of strategy names.  By default this is set to None and we use all strategies.
            plot: If set to True, display plots of equity, drawdowns and returns.  Default False
            float_precision: Number of significant figures to show in returns.  Default 4
        '''
        returns = self.df_returns(sampling_freq, strategy_names)
        ev = compute_return_metrics(returns.timestamp.values, returns.ret.values, returns.equity.values[0])
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
        timestamps = returns.timestamp.values
        ev = compute_return_metrics(timestamps, returns.ret.values, returns.equity.values[0])
        plot_return_metrics(ev.metrics())
        
    def __repr__(self):
        return f'{self.name} {self.strategies.keys()}'
    
if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags = doctest.NORMALIZE_WHITESPACE)

