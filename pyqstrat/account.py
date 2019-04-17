#cell 0
from collections import defaultdict, deque
import math
import pandas as pd
import numpy as np
from copy import copy
from pyqstrat.pq_utils import str2date
from pyqstrat.pq_types import ContractGroup

#cell 1
def _calc_pnl(open_trades, new_trades, ending_close, multiplier):
    '''
    >>> from collections import deque
    >>> from pyqstrat.pq_types import Trade
    >>> from pyqstrat.pq_types import Contract, ContractGroup
    >>> contract_group = ContractGroup.create('IBM')
    >>> ibm = Contract.create('IBM', contract_group = contract_group)
    >>> trades = deque([Trade(ibm, np.datetime64('2018-01-01 10:15:00'), 3, 51.),
    ...              Trade(ibm, np.datetime64('2018-01-01 10:20:00'), 10, 50.),
    ...              Trade(ibm, np.datetime64('2018-01-02 11:20:00'), -5, 45.)])
    >>> print(_calc_pnl(open_trades = deque(), new_trades = trades, ending_close = 54, multiplier = 100))
    (deque([IBM 2018-01-01 10:20:00 qty: 8 prc: 50 order: None]), 3200.0, -2800.0)
    >>> trades = deque([Trade(ibm, np.datetime64('2018-01-01 10:15:00'), -8, 10.),
    ...          Trade(ibm, np.datetime64('2018-01-01 10:20:00'), 9, 11.),
    ...          Trade(ibm, np.datetime64('2018-01-02 11:20:00'), -4, 6.)])
    >>> print(_calc_pnl(open_trades = deque(), new_trades = trades, ending_close = 5.8, multiplier = 100))
    (deque([IBM 2018-01-02 11:20:00 qty: -3 prc: 6 order: None]), 60.00000000000006, -1300.0)
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
        self.contract = contract
        self.timestamps = timestamps
        self._price_function = price_function
        self.strategy_context = strategy_context
        self._unrealized = np.full(len(self.timestamps), np.nan, dtype = np.float); self._unrealized[0] = 0
        self._realized = np.full(len(self.timestamps), np.nan, dtype = np.float); self._realized[0] = 0
        
        self._commission = np.zeros(len(self.timestamps), dtype = np.float);
        self._fee = np.zeros(len(self.timestamps), dtype = np.float);
        
        self._net_pnl = np.full(len(self.timestamps), np.nan, dtype = np.float); self._net_pnl[0] = 0
        self._position = np.full(len(self.timestamps), np.nan, dtype = np.float); self._position[0] = 0
        self._price = np.full(len(self.timestamps), np.nan, dtype = np.float)
        self._trades = []
        self.open_trades = deque()
        self.first_calc = True
        
    def add_trades(self, trades):
        '''Args:
            trades: A list of Trade objects
        '''
        self._trades += trades
        
    def calc(self, prev_i, i, account_timestamps):
        '''Compute pnl and store it internally
        
        Args:
            prev_i: Start index to compute pnl from
            i: End index to compute pnl to
        '''
        assert(i >= 0 and prev_i >= 0)
        
        if self.first_calc:
            self._unrealized[:prev_i + 1] = 0
            self._realized[:prev_i + 1] = 0
            self._net_pnl [:prev_i + 1] = 0
            self._position[:prev_i + 1] = 0
            self.first_calc = False


        calc_trades = deque([trade for trade in self._trades if (prev_i == 0 or trade.timestamp > self.timestamps[prev_i])
                             and trade.timestamp <= self.timestamps[i]])
        trade_qty = sum([trade.qty for trade in calc_trades])
        
        self._position[i] = self._position[prev_i] + trade_qty
        
        price = self._price_function(self.contract, account_timestamps, i, self.strategy_context)
        
        if not np.isfinite(price):
            unrealized = self._unrealized[prev_i]
            realized = 0. 
        else:
            open_trades, unrealized, realized = _calc_pnl(self.open_trades, calc_trades, price, self.multiplier)
            self.open_trades = open_trades
            
        self._unrealized[i] = unrealized
        self._realized[i] = self._realized[prev_i] + realized
        #print(f'new pos: {self._position[i]} old pos: {self._position[prev_i]} prev_i: {prev_i} i: {i} offset: {self.offset} trade_qty: {trade_qty}')
        self._commission[i] = self._commission[prev_i] + sum([trade.commission for trade in calc_trades])
        self._fee[i] = self._fee[prev_i] + sum([trade.fee for trade in calc_trades])
        self._net_pnl[i] = self._realized[i] + self._unrealized[i] - self._commission[i] - self._fee[i]
        self._price[i] = price
        if np.isnan(self._net_pnl[i]):
            raise Exception(f'net_pnl: nan i: {i} prev_i: {prev_i} realized: {self._realized[i]} unrealized: {self._unrealized[i]} commission: ' +
                            f'{self._commission[i]} fee: {self._fee[i]}')
        
    def trades(self, start_date = None, end_date = None):
        '''Get a list of trades
        
        Args:
            start_date: A string or numpy datetime64.  Trades with trade timestamps >= start_date will be returned.  Default None
            end_date: A string or numpy datetime64.  Trades with trade timestamps <= end_date will be returned.  Default None
        '''
        start_date, end_date = str2date(start_date), str2date(end_date)
        trades = [trade for trade in self._trades if (start_date is None or trade.timestamp >= start_date) and (
            end_date is None or trade.timestamp <= end_date)]
        return trades
    
    def position(self, i):
        return self._position[i]
    
    def net_pnl(self, i):
        return self._net_pnl[i]
         
    def df(self):
        '''Returns a pandas dataframe with pnl data, indexed by date'''
        mask = (self._position != 0) & np.isfinite(self._unrealized) & np.isfinite(self._realized)
        df = pd.DataFrame({'timestamp' : self.timestamps[mask], 
                           'unrealized' : self._unrealized[mask],
                           'realized' : self._realized[mask], 
                           'commission' : self._commission[mask],
                           'fee' : self._fee[mask], 
                           'net_pnl' : self._net_pnl[mask],
                           'position' : self._position[mask],
                           'price' : self._price[mask]})
        df['symbol'] = self.symbol
        return df[['symbol', 'timestamp', 'unrealized', 'realized', 'commission', 'fee', 'net_pnl', 'position', 'price']]
    
def _get_calc_indices(timestamps):
    calc_timestamps = np.unique(timestamps.astype('M8[D]'))
    calc_indices = np.searchsorted(timestamps, calc_timestamps, side='left') - 1
    if calc_indices[0] == -1: calc_indices[0] = 0
    return calc_indices

def leading_nan_to_zero(df, columns):
    for column in columns:
        vals = df[column].values
        first_non_nan_index = np.ravel(np.nonzero(~np.isnan(vals)))
        if len(first_non_nan_index):
            first_non_nan_index = first_non_nan_index[0]
        else:
            first_non_nan_index = -1

        if first_non_nan_index > 0 and first_non_nan_index < len(vals):
            vals[:first_non_nan_index] = np.nan_to_num(vals[:first_non_nan_index])
            df[column] = vals
    return df

class Account:
    '''An Account calculates pnl for a set of contracts'''
    def __init__(self, contract_groups, timestamps, price_function, strategy_context, starting_equity = 1.0e6, calc_frequency = 'D'):
        '''
        Args:
            contract_groups (list of :obj:`ContractGroup`): Contract groups that we want to compute PNL for
            timestamps (list of np.datetime64): Timestamps that we might compute PNL at
            price_function (function): Function that takes a symbol, timestamps, index, strategy context and 
                returns the price used to compute pnl
            starting_equity (float, optional): Starting equity in account currency.  Default 1.e6
            calc_frequency (str, optional): Account will calculate pnl at this frequency.  Default 'D' for daily
        '''
        if calc_frequency != 'D': raise Exception('unknown calc frequency: {}'.format(calc_frequency))
        self.calc_freq = calc_frequency
        self.current_calc_index = 0
        self.starting_equity = starting_equity
        self._price_function = price_function
        self.strategy_context = strategy_context
        
        self.timestamps = timestamps
        self.calc_indices = _get_calc_indices(timestamps)
        
        self._equity = np.full(len(timestamps), np.nan, dtype = np.float); 
        self._equity[0] = self.starting_equity
        
        self.contracts = set()
        
        self.symbol_pnls = {}
        self.symbols = set()
        
    def symbols(self):
        return self.symbols
        
    def _add_contract(self, contract, timestamp):
        self.symbol_pnls[contract.symbol] = ContractPNL(contract, self.timestamps, self._price_function, self.strategy_context)
        self.symbols.add(contract.symbol)
        self.contracts.add(contract)
        
    def add_trades(self, trades):
        trades = sorted(trades, key = lambda x : getattr(x, 'timestamp'))
        for trade in trades:
            contract = trade.contract
            if contract.symbol not in self.symbols: self._add_contract(contract, trade.timestamp)
            self.symbol_pnls[trade.contract.symbol].add_trades([trade])
        
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
            
        if not len(self.symbol_pnls):
            prev_equity = self._equity[self.current_calc_index]
            for idx in intermediate_calc_indices:
                self._equity[calc_indices[idx]] = prev_equity
            self.current_calc_index = i
            return
        
        prev_calc_index = self.current_calc_index
        for idx in intermediate_calc_indices:
            calc_index = calc_indices[idx]
            self._equity[calc_index] = self._equity[prev_calc_index]
            for symbol, symbol_pnl in self.symbol_pnls.items():
                symbol_pnl.calc(prev_calc_index, calc_index, self.timestamps)
                net_pnl_diff = symbol_pnl.net_pnl(calc_index) - symbol_pnl.net_pnl(prev_calc_index)
                if np.isfinite(net_pnl_diff): self._equity[calc_index] += net_pnl_diff
                if symbol == "XXXXXX":
                    print(f'prev_calc_index: {prev_calc_index} calc_index: {calc_index} prev_equity:' + 
                          f' {self._equity[prev_calc_index]} curr_equity: {self._equity[calc_index]}' + 
                          f' net_pnl: {symbol_pnl.net_pnl(calc_index)}' + 
                          f' prev_net_pnl: {symbol_pnl.net_pnl(prev_calc_index)}')
            prev_calc_index = calc_index
        self.current_calc_index = i
        
    def position(self, contract_group, timestamp):
        '''Returns position for a contract_group at a given date in number of contracts or shares.  
            Will cause calculation if Account has not previously calculated up to this date'''
        i = np.searchsorted(self.timestamps, timestamp)
        if i == len(self.timestamps) or self.timestamps[i] != timestamp:
            raise Exception(f'Invalid timestamp: {timestamp}')
        self.calc(i)
        position = 0
        for contract in contract_group.contracts:
            symbol = contract.symbol
            if symbol not in self.symbol_pnls: continue
            position += self.symbol_pnls[symbol].position(i)
        return position
    
    def positions(self, contract_group, timestamp):
        '''
        Returns all non-zero positions in a contract group
        '''
        i = np.searchsorted(self.timestamps, timestamp)
        self.calc(i)
        positions = []
        for contract in contract_group.contracts:
            symbol = contract.symbol
            if symbol not in self.symbol_pnls: continue
            position = self.symbol_pnls[symbol].position(i)
            if not math.isclose(position, 0): positions.append((contract, position))
        return positions
    
    def equity(self, timestamp):
        '''Returns equity in this account in Account currency.  Will cause calculation if Account has not previously 
            calculated up to this date'''
        i = np.searchsorted(self.timestamps, timestamp)
        self.calc(i)
        return self._equity[i]
    
    def trades(self, contract_group = None, start_date = None, end_date = None):
        '''Returns a list of trades with the given symbol and with trade date between (and including) start date 
            and end date if they are specified.
            If symbol is None trades for all symbols are returned'''
        start_date, end_date = str2date(start_date), str2date(end_date)
        if contract_group is None:
            trades = []
            for symbol, sym_pnl in self.symbol_pnls.items():
                trades += sym_pnl.trades(start_date, end_date)
            return trades
        else:
            trades = []
            for contract in contract_group.contracts:
                symbol = contract.symbol
                if symbol not in self.symbol_pnls: continue
                trades += self.symbol_pnls[symbol].trades(start_date, end_date)
            return trades
               
    def transfer_cash(self, date, amount):
        '''Move cash from one portfolio to another'''
        i = np.searchsorted(self.timestamps, date)
        curr_equity = self.equity(date)
        if (amount > curr_equity): amount = curr_equity # Cannot make equity negative
        self._equity[i] -= amount
        return amount

    def df_pnl(self, contract_groups = None):
        '''Returns a dataframe with P&L columns.
        Args:
            contract_group (:obj:`ContractGroup`, optional): Return PNL for this contract group.  
                If None (default), include all contract groups
        '''
        
        if contract_groups is None: 
            contract_groups = set([contract.contract_group for contract in self.contracts])
        
        if isinstance(contract_groups, ContractGroup): contract_groups = [contract_groups]
        
        dfs = []
        for contract_group in contract_groups:
            for contract in contract_group.contracts:
                symbol = contract.symbol
                if symbol not in self.symbol_pnls: continue
                df = self.symbol_pnls[symbol].df()
                df['contract_group'] = contract_group.name
                dfs.append(df)
        ret_df = pd.concat(dfs)
        ret_df = ret_df[['timestamp', 'contract_group', 'symbol', 'position', 'price', 'unrealized', 'realized', 
                         'commission', 'fee', 'net_pnl']]
        df_equity = pd.DataFrame({'timestamp' : self.timestamps, 'equity' : self._equity}).dropna()
        ret_df = pd.merge(ret_df, df_equity, on = ['timestamp'], how = 'outer')
        ret_df = ret_df.sort_values(by = ['timestamp', 'contract_group', 'symbol'])
        ret_df = leading_nan_to_zero(ret_df, ['unrealized', 'realized', 'commission', 'fee', 'net_pnl'])
        
        return ret_df[['timestamp', 'contract_group', 'symbol', 'position', 'price', 'unrealized', 'realized', 
                       'commission', 'fee', 'net_pnl', 'equity']]
    
    def df_trades(self, contract_group = None, start_date = None, end_date = None):
        '''Returns a dataframe of trades
        Args:
            contract_group (:obj:`ContractGroup`, optional): Return trades for this contract group.  If None (default), include all contract groups
            start_date (:obj:`np.datetime64`, optional): Include trades with date greater than or equal to this timestamp.
            end_date (:obj:`np.datetime64`, optional): Include trades with date less than or equal to this timestamp.
        '''
        start_date, end_date = str2date(start_date), str2date(end_date)
        if contract_group:
            trades = []
            for contract in contract_group.contracts:
                symbol = contract.symbol
                if symbol not in self.symbol_pnls: continue
                trades += self.symbol_pnls[symbol].trades(start_date, end_date)
        else:
            trades = [v.trades(start_date, end_date) for v in self.symbol_pnls.values()]
            trades = [trade for sublist in trades for trade in sublist] # flatten list
        df = pd.DataFrame.from_records([(trade.contract.symbol, trade.timestamp, trade.qty, trade.price, 
                                         trade.fee, trade.commission, trade.order.timestamp, trade.order.qty, 
                                         trade.order.reason_code, 
                                         (trade.order.properties.__dict__ if trade.order.properties.__dict__ else ''), 
                                         (trade.contract.properties.__dict__ if trade.contract.properties.__dict__ else '')
                                        ) for trade in trades],
                    columns = ['symbol', 'timestamp', 'qty', 'price', 'fee', 'commission', 'order_date', 'order_qty', 
                               'reason_code', 'order_props', 'contract_props'])
        df = df.sort_values(by = ['timestamp', 'symbol'])
        return df
    
#def test_account():
if __name__ == "__main__":


    from pyqstrat.pq_types import Contract, ContractGroup, Trade
    from pyqstrat.orders import MarketOrder
    import math

    def get_close_price(contract, timestamps, idx, strategy_context):
        if contract.symbol == "IBM":
            price = idx + 10.1
        elif contract.symbol == "MSFT":
            price = idx + 15.3
        else:
            raise Exception(f'unknown contract: {contract}')
        return price
    
    ibm_cg = ContractGroup.create('IBM')
    msft_cg = ContractGroup.create('MSFT')
    
    ibm_contract = Contract.create('IBM', contract_group = ibm_cg)
    msft_contract = Contract.create('MSFT', contract_group  = msft_cg)
    timestamps = np.array(['2018-01-01 09:00', '2018-01-02 08:00', '2018-01-02 09:00', '2018-01-05 13:35'], dtype = 'M8[m]')
    account = Account([ibm_cg, msft_cg], timestamps, get_close_price, None)
    #account = Account([Contract(symbol)], timestamps, get_close_price)
    trade_1 = Trade(ibm_contract, np.datetime64('2018-01-02 08:00'), 10, 10.1, 0.01, 
                    order = MarketOrder(ibm_contract, np.datetime64('2018-01-01 08:55'), 10))
    trade_2 = Trade(ibm_contract, np.datetime64('2018-01-04 13:55'), 20, 15.1, 0.02, 
                    order = MarketOrder(ibm_contract, np.datetime64('2018-01-03 13:03'), 20))
    trade_3 = Trade(msft_contract, timestamps[1], 20, 13.2, 0.04, 
                    order = MarketOrder(msft_contract, timestamps[1], 15))
    trade_4 = Trade(msft_contract, timestamps[2], 20, 16.2, 0.05, 
                    order = MarketOrder(msft_contract, timestamps[2], 20))

    account.add_trades([trade_1, trade_2, trade_3, trade_4])
    np.set_printoptions(formatter = {'float' : lambda x : f'{x:.4f}'})  # After numpy 1.13 positive floats don't have a leading space for sign
    account.calc(3)
    account.df_trades()
    assert(len(account.df_trades()) == 4)
    assert(len(account.df_pnl()) == 5)
    assert(np.allclose(np.array([1000000, 1000123.9, 1000123.9, 1000133.88,1000133.88]
                               ), account.df_pnl().equity.values, rtol = 0))
    assert(np.allclose(np.array([10, 40, 30, 40]), account.df_pnl().position.values[1:], rtol = 0))
    assert(np.allclose(np.array([1000000.0000, 1000123.9000, 1000133.8800]), account.df_pnl([ibm_cg]).equity.values, rtol = 0))
    
if __name__ == "__mainx__":
    test_account()
    import doctest
    doctest.testmod(optionflags = doctest.NORMALIZE_WHITESPACE)

#cell 2


#cell 3


