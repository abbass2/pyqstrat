#cell 0
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from copy import copy
from pyqstrat.pq_utils import str2date

#cell 1
def _calc_pnl(open_trades, new_trades, ending_close, multiplier):
    '''
    >>> from collections import deque
    >>> from pyqstrat.pq_types import Trade
    >>> trades = deque([Trade('IBM', np.datetime64('2018-01-01 10:15:00'), 3, 51.),
    ...              Trade('IBM', np.datetime64('2018-01-01 10:20:00'), 10, 50.),
    ...              Trade('IBM', np.datetime64('2018-01-02 11:20:00'), -5, 45.)])
    >>> print(_calc_pnl(open_trades = deque(), new_trades = trades, ending_close = 54, multiplier = 100))
    (deque([IBM 2018-01-01 10:20 qty: 8 prc: 50 order: None]), 3200.0, -2800.0)
    >>> trades = deque([Trade('IBM', np.datetime64('2018-01-01 10:15:00'), -8, 10.),
    ...          Trade('IBM', np.datetime64('2018-01-01 10:20:00'), 9, 11.),
    ...          Trade('IBM', np.datetime64('2018-01-02 11:20:00'), -4, 6.)])
    >>> print(_calc_pnl(open_trades = deque(), new_trades = trades, ending_close = 5.8, multiplier = 100))
    (deque([IBM 2018-01-02 11:20 qty: -3 prc: 6 order: None]), 60.00000000000006, -1300.0)
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
        
class ContractPNL:
    '''Computes pnl for a single contract over time given trades and market data'''
    def __init__(self, contract, timestamps, price_function, strategy_context):
        self.symbol = contract.symbol
        self.multiplier = contract.multiplier
        self.timestamps = timestamps
        self.price_function = price_function
        self.strategy_context = strategy_context
        self.unrealized = np.full(len(self.timestamps), np.nan, dtype = np.float); self.unrealized[0] = 0
        self.realized = np.full(len(self.timestamps), np.nan, dtype = np.float); self.realized[0] = 0
        
        self.commission = np.zeros(len(self.timestamps), dtype = np.float);
        self.fee = np.zeros(len(self.timestamps), dtype = np.float);
        
        self.net_pnl = np.full(len(self.timestamps), np.nan, dtype = np.float); self.net_pnl[0] = 0
        self.position = np.full(len(self.timestamps), np.nan, dtype = np.float); self.position[0] = 0
        self.price = np.full(len(self.timestamps), np.nan, dtype = np.float)
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
        calc_trades = deque([trade for trade in self._trades if trade.timestamp > self.timestamps[prev_i] 
                             and trade.timestamp <= self.timestamps[i]])
        
        price = self.price_function(self.symbol, self.timestamps, i, self.strategy_context)
        
        if not np.isfinite(price):
            unrealized = self.unrealized[prev_i]
            realized = 0. 
        else:
            open_trades, unrealized, realized = _calc_pnl(self.open_trades, calc_trades, price, self.multiplier)
            self.open_trades = open_trades
            
        self.unrealized[i] = unrealized
        self.realized[i] = self.realized[prev_i] + realized
        trade_qty = sum([trade.qty for trade in calc_trades])
        self.position[i] = self.position[prev_i] + trade_qty
        self.commission[i] = self.commission[prev_i] + sum([trade.commission for trade in calc_trades])
        self.fee[i] = self.fee[prev_i] + sum([trade.fee for trade in calc_trades])
        self.net_pnl[i] = self.realized[i] + self.unrealized[i] - self.commission[i] - self.fee[i]
        self.price[i] = price
        if np.isnan(self.net_pnl[i]):
            raise Exception(f'net_pnl: nan i: {i} realized: {self.realized[i]} unrealized: {self.unrealized[i]} commission: {self.commission[i]} fee: {self.fee[i]}')
        
    def trades(self, start_date = None, end_date = None):
        '''Get a list of trades
        
        Args:
            start_date: A string or numpy datetime64.  Trades with trade timestamps >= start_date will be returned.  Default None
            end_date: A string or numpy datetime64.  Trades with trade timestamps <= end_date will be returned.  Default None
        '''
        start_date, end_date = str2date(start_date), str2date(end_date)
        trades = [trade for trade in self._trades if (start_date is None or trade.timestamp >= start_date) and (end_date is None or trade.timestamp <= end_date)]
        return trades
         
    def df(self):
        '''Returns a pandas dataframe with pnl data, indexed by date'''
        df = pd.DataFrame({'timestamp' : self.timestamps, 'unrealized' : self.unrealized, 'realized' : self.realized, 
                           'fee' : self.fee, 'net_pnl' : self.net_pnl, 'position' : self.position, 'price' : self.price})
        df.dropna(subset = ['unrealized', 'realized'], inplace = True)
        df['symbol'] = self.symbol
        return df[['symbol', 'timestamp', 'unrealized', 'realized', 'fee', 'net_pnl', 'position', 'price']]
    
def _get_calc_indices(timestamps):
    calc_timestamps = np.unique(timestamps.astype('M8[D]'))
    calc_indices = np.searchsorted(timestamps, calc_timestamps, side='left') - 1
    if calc_indices[0] == -1: calc_indices[0] = 0
    return calc_indices

class Account:
    '''An Account calculates pnl for a set of contracts'''
    def __init__(self, contracts, timestamps, price_function, strategy_context, starting_equity = 1.0e6, calc_frequency = 'D'):
        '''
        Args:
            contracts (list of Contract): Contracts that we want to compute PNL for
            timestamps (list of np.datetime64): Timestamps that we might compute PNL at
            price_function (function): Function that takes a symbol, timestamps, index, strategy context and 
                returns the price used to compute pnl
            starting_equity (float, optional): Starting equity in account currency.  Default 1.e6
            calc_frequency (str, optional): Account will calculate pnl at this frequency.  Default 'D' for daily
        '''
        if calc_frequency != 'D': raise Exception('unknown calc frequency: {}'.format(calc_frequency))
        self.calc_freq = calc_frequency
        self.contract_pnls = defaultdict()
        self.current_calc_index = 0
        self.starting_equity = starting_equity
        self.price_function = price_function
        self.strategy_context = strategy_context
        
        contract_symbols = sorted([contract.symbol for contract in contracts])
        
        self.timestamps = timestamps
        self.calc_indices = _get_calc_indices(timestamps)
        
        self._equity = np.full(len(timestamps), np.nan, dtype = np.float); 
        self._equity[0] = self.starting_equity

        if contracts is not None:
            for contract in contracts: 
                self.add_contract(contract)
        
    def symbols(self):
        return list(self.contract_pnls.keys())
        
    def add_contract(self, contract):
         self.contract_pnls[contract.symbol] = ContractPNL(contract, self.timestamps, self.price_function, self.strategy_context)
        
    def _add_trades(self, symbol, trades):
        self.contract_pnls[symbol].add_trades(trades)
        
    def calc(self, i):
        '''
        Computes P&L and stores it internally for all contracts.
        
        Args:
            i: Index to compute P&L at.  Account remembers the last index it computed P&L up to and will compute P&L
                between these two indices.  If there is more than one day between the last index and current index, we will 
                include pnl for end of day at those dates as well.
        '''
        calc_indices = self.calc_indices[:]
        if self.current_calc_index == i: return
        # Find the last timestamp per day that is between the previous index we computed and the current index,
        # so we can compute daily pnl in addition to the current index pnl
        intermediate_calc_indices = np.where((calc_indices > self.current_calc_index) & (calc_indices <= i))
        # The previous operations gives us a nested array with one element so flatten it to a 1-D array
        intermediate_calc_indices = np.ravel(intermediate_calc_indices)

        if not len(intermediate_calc_indices) or calc_indices[intermediate_calc_indices[-1]] != i: 
            calc_indices = np.append(calc_indices, i)
            intermediate_calc_indices = np.append(intermediate_calc_indices, len(calc_indices) - 1)
            
        for symbol, symbol_pnl in self.contract_pnls.items():
            prev_calc_index = self.current_calc_index
            for idx in intermediate_calc_indices:
                calc_index = calc_indices[idx]
                symbol_pnl.calc(prev_calc_index, calc_index)
                self._equity[calc_index] = self._equity[prev_calc_index] + \
                    symbol_pnl.net_pnl[calc_index] - symbol_pnl.net_pnl[prev_calc_index]
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
        '''Returns equity in this account in Account currency.  Will cause calculation if Account has not previously 
            calculated up to this date'''
        i = self.find_index_before(date)
        self.calc(i)
        return self._equity[i]
    
    def trades(self, symbol = None, start_date = None, end_date = None):
        '''Returns a list of trades with the given symbol and with trade date between (and including) start date 
            and end date if they are specified.
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
        return np.searchsorted(self.timestamps, date)
        
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
            ret = ret.groupby('timestamp').sum()
        df_equity = pd.DataFrame({'timestamp' : self.timestamps, 'equity' : self._equity}).dropna()
        ret = pd.merge(ret, df_equity, on = ['timestamp'], how = 'outer')
        return ret.reset_index()
    
    def df_trades(self, symbol = None, start_date = None, end_date = None):
        '''Returns a dataframe with data from trades with the given symbol and with trade date between (and including) 
            start date and end date if they are specified.  If symbol is None, trades for all symbols are returned'''
        start_date, end_date = str2date(start_date), str2date(end_date)
        if symbol:
            trades = self.contract_pnls[symbol].trades(start_date, end_date)
        else:
            trades = [v.trades(start_date, end_date) for v in self.contract_pnls.values()]
            trades = [trade for sublist in trades for trade in sublist] # flatten list
        df = pd.DataFrame.from_records([(trade.symbol, trade.timestamp, trade.qty, trade.price, 
                                         trade.fee, trade.commission, trade.order.timestamp, trade.order.qty, trade.order.params()
                                        ) for trade in trades],
                    columns = ['symbol', 'timestamp', 'qty', 'price', 'fee', 'commission', 'order_date', 'order_qty', 'order_params'])
        return df
    
def test_account():
    from pyqstrat.pq_types import Contract, Trade
    from pyqstrat.orders import MarketOrder
    import math

    def get_close_price(symbol, timestamps, idx, strategy_context):
        indices = np.arange(len(timestamps))
        prices = indices + 10
        price = dict(zip(indices, prices))[idx]
        return price
    
    symbol = 'IBM'
    timestamps = np.array(['2018-01-01 05:35', '2018-01-02 08:00', '2018-01-02 09:00', '2018-01-05 13:35'], dtype = 'M8[m]')
    account = Account([Contract(symbol)], timestamps, get_close_price, None)
    #account = Account([Contract(symbol)], timestamps, get_close_price)
    trade_1 = Trade(symbol, np.datetime64('2018-01-01 09:00'), 10, 10.1, 0.01, 
                    order = MarketOrder(symbol, np.datetime64('2018-01-01 08:55'), 10))
    trade_2 = Trade(symbol, np.datetime64('2018-01-04 13:55'), 20, 15.1, 0.02, 
                    order = MarketOrder(symbol, np.datetime64('2018-01-03 13:03'), 20))
    account._add_trades('IBM', [trade_1, trade_2])
    np.set_printoptions(formatter = {'float' : lambda x : f'{x:.4f}'})  # After numpy 1.13 positive floats don't have a leading space for sign
    account.calc(3)
    account.df_trades()
    assert(len(account.df_trades()) == 2)
    assert(len(account.df_pnl()) == 3)
    assert(np.allclose(np.array([1000000, 1000018.99, 999986.97]), account.df_pnl().equity.values, rtol = 0))
    assert(np.allclose(np.array([0, 10, 30]), account.df_pnl().position.values, rtol = 0))
    assert(np.allclose(np.array([0, 10, 30]), account.df_pnl().position.values, rtol = 0))
    
if __name__ == "__main__":
    test_account()

