#cell 0
from collections import defaultdict, deque
import math
import pandas as pd
import numpy as np
from copy import copy
from pyqstrat.pq_utils import str2date
from pyqstrat.pq_types import ContractGroup

#cell 1
def _calc_trade_pnl(open_qtys, open_prices, new_qtys, new_prices, multiplier):
    '''
    >>> print(_calc_trade_pnl(open_qtys = np.array([]), open_prices = np.array([]), new_qtys = np.array([-8, 9, -4]), new_prices = np.array([10, 11, 6]),
    ...          multiplier = 100))
    (array([-3]), array([6]), -3.0, 6.0, -1300.0)
    >>> print(_calc_trade_pnl(open_qtys = np.array([]), open_prices = np.array([]), new_qtys = np.array([3, 10, -5]), new_prices = np.array([51, 50, 45]),
    ...          multiplier = 100))
    (array([8]), array([50]), 8.0, 50.0, -2800.0)
    >>> print(_calc_trade_pnl(open_qtys = np.array([]), open_prices = np.array([]), 
    ...                new_qtys = np.array([-58, -5, -5, 6, -8, 5, 5, -5, 19, 7, 5, -5, 39]),
    ...                new_prices = np.array([2080, 2075.25, 2070.75, 2076, 2066.75, 2069.25, 2074.75, 2069.75, 2087.25, 2097.25, 2106, 2088.25, 2085.25]),
    ...                multiplier = 50))
    (array([], dtype=float64), array([], dtype=float64), 0.0, nan, -33762.5)    '''
    
    realized = 0.
    unrealized = 0.
    
    open_qtys = open_qtys.copy()
    new_qtys = new_qtys.copy()
    open_prices = open_prices.copy()
    new_prices = new_prices.copy()
    
    # Try to net all new trades against existing non-netted trades.
    # Append any remaining non-netted new trades to end of existing trades
    while not all(new_qtys == 0):
        # Always try to net first non-zero new trade against first non-zero existing trade
        # FIFO acccounting
        new_idx = np.nonzero(new_qtys)[0][0]
        new_qty, new_price = new_qtys[new_idx], new_prices[new_idx]
        open_idx_array = np.nonzero(open_qtys)[0]
        
        if len(open_idx_array): 
            open_idx = open_idx_array[0]
            open_qty, open_price = open_qtys[open_idx], open_prices[open_idx]
            
            if math.copysign(1, open_qty) == math.copysign(1, new_qty):
                # Nothing to net against so add this trade to the deque and wait for the next offsetting trade
                open_qtys = np.append(open_qtys, new_qty)
                open_prices = np.append(open_prices, new_price)
                new_qtys[new_idx] = 0

            elif abs(new_qty) > abs(open_qty):
                # New trade has more qty than offsetting trade so:
                # a. net against offsetting trade
                # b. remove the offsetting trade
                # c. reduce qty of new trade
                open_qty, open_price = open_qtys[open_idx], open_prices[open_idx]
                realized += open_qty * (new_price - open_price)
                open_qtys[open_idx] = 0
                new_qtys[new_idx] += open_qty
            else:
                # New trade has less qty than offsetting trade so:
                # a. net against offsetting trade
                # b. remove new trade
                # c. reduce qty of offsetting trade
                realized += new_qty * (open_price - new_price)
                new_qtys[new_idx] = 0
                open_qtys[open_idx] += new_qty
                
        else:

            # Nothing to net against so add this trade to the open trades queue and wait for the next offsetting trade
            open_qtys = np.append(open_qtys, new_qty)
            open_prices = np.append(open_prices, new_price)
            new_qtys[new_idx] = 0
            
    mask = open_qtys != 0
    open_qtys = open_qtys[mask]
    open_prices = open_prices[mask]
    open_qty = np.sum(open_qtys)
    if math.isclose(open_qty, 0):
        weighted_avg_price = np.nan
    else:
        weighted_avg_price = np.sum(open_qtys * open_prices) / open_qty
        
    return open_qtys, open_prices, open_qty, weighted_avg_price, realized * multiplier
    
def find_last_non_nan_index(array):
    i = np.nonzero(np.isfinite(array))[0]
    if len(i): return i[-1]
    return 0

class ContractPNL:
    '''Computes pnl for a single contract over time given trades and market data'''
    def __init__(self, contract, account_timestamps, price_function, strategy_context):
        self.symbol = contract.symbol
        self.multiplier = contract.multiplier
        self.contract = contract
        
        self._price_function = price_function
        self.strategy_context = strategy_context
        
        self.account_timestamps = account_timestamps
        self._timestamps = None
        
        # Store trades that are not offset so when new trades come in we can offset against these to calc pnl
        self.open_qtys = np.empty(0, dtype = np.int)
        self.open_prices = np.empty(0, dtype = np.float)
        
        self.last_trade_calc_idx = 0
        self.last_net_pnl_calc_idx = 0
 
        self._trade_timestamps = None
        
    def _add_trades(self, trades):
        '''
        Args:
            trades (list of :obj:`Trade`): Must be sorted by timestamp
        '''
        num_trades = len(trades)
        
        if not num_trades: return
        # Trades should already be sorted by timestamp
        timestamps = np.empty(num_trades, dtype = trades[0].timestamp.dtype)
        qtys = np.empty(num_trades, dtype = np.int)
        prices = np.empty(num_trades, dtype = np.float)
        commissions =  np.empty(num_trades, dtype = np.float)
        fees = np.empty(num_trades, dtype = np.float)
        
        for i, trade in enumerate(trades):
            timestamps[i] = trade.timestamp
            qtys[i] = trade.qty
            prices[i] = trade.price
            commissions[i] =  trade.commission
            fees[i] = trade.fee
            
        self.calc_trades(timestamps, qtys, prices, commissions, fees)
            
    def calc_trades(self, timestamps, qtys, prices, commissions, fees):
        start_timestamp, end_timestamp = timestamps[0], timestamps[-1]
        self._expand_arrays(start_timestamp, end_timestamp)
        
        prev_idx = self.last_trade_calc_idx
         
        open_qtys, open_prices, open_qty, weighted_avg_price, realized = _calc_trade_pnl(self.open_qtys, self.open_prices, qtys, prices, self.multiplier)
       
        self.open_qtys = open_qtys
        self.open_prices = open_prices
        
        curr_idx = np.searchsorted(self.account_timestamps, end_timestamp)
        assert(curr_idx < len(self.account_timestamps) and self.account_timestamps[curr_idx] == end_timestamp)
        curr_idx -= self.offset

        if curr_idx <= prev_idx: return
        
        self._position[curr_idx] = self._position[prev_idx] + np.sum(qtys)
        self._realized[curr_idx] = self._realized[prev_idx] + realized
        self._commission[curr_idx] = self._commission[prev_idx] + np.sum(commissions)
        self._fee[curr_idx] = self._fee[prev_idx] + np.sum(fees)
        
        self._open_qty[curr_idx] = open_qty
        self._weighted_avg_price[curr_idx] = weighted_avg_price
 
        self.last_trade_calc_idx = curr_idx
    
        self.calc_net_pnl(curr_idx + self.offset)
        
    def _expand_arrays(self, start_timestamp, end_timestamp):

        start_idx = -1
        if start_timestamp is not None:
            start_idx = np.searchsorted(self.account_timestamps, start_timestamp)
            assert(start_idx < len(self.account_timestamps) and self.account_timestamps[start_idx] == start_timestamp)
        end_idx = np.searchsorted(self.account_timestamps, end_timestamp)
        assert(end_idx < len(self.account_timestamps) and self.account_timestamps[end_idx] == end_timestamp)
        
        if self._timestamps is None: # First time we are called
            assert(start_idx != 0)  # First time we cannot be called with index 0 since trade must occur after at least one bar
            size = end_idx - start_idx + 2 # Create a zero row at the beginning
            self._timestamps = self.account_timestamps[start_idx - 1: end_idx + 1]
            self._unrealized = np.full(size, np.nan, dtype = np.float); self._unrealized[0] = 0
            self._realized = np.full(size, np.nan, dtype = np.float); self._realized[0] = 0
            self._commission = np.full(size, np.nan, dtype = np.float); self._commission[0] = 0
            self._fee = np.full(size, np.nan, dtype = np.float); self._fee[0] = 0
            self._net_pnl = np.full(size, np.nan, dtype = np.float); self._net_pnl[0] = 0
            self._position = np.full(size, np.nan, dtype = np.float); self._position[0] = 0
            self._price = np.full(size, np.nan, dtype = np.float); self._price[0] = np.nan
            self._open_qty = np.full(size, np.nan, dtype = np.float); self._open_qty[0] = 0
            self._weighted_avg_price = np.full(size, np.nan, dtype = np.float); self._weighted_avg_price[0] = np.nan
            
            self.offset = start_idx - 1
            assert(self.offset >= 0)
            self.size = size
            
        elif end_idx - self.offset >= self.size: # Expand arrays
            size = end_idx - self.offset - self.size + 1
            if size <= 0:
                 return
            new_timestamps_start = np.searchsorted(self.account_timestamps, self._timestamps[-1])
            new_timestamps = self.account_timestamps[new_timestamps_start + 1:end_idx + 1]
            self._timestamps = np.concatenate((self._timestamps, new_timestamps))
            self._unrealized = np.concatenate((self._unrealized, np.full(size, np.nan, dtype = np.float)))
            self._realized = np.concatenate((self._realized, np.full(size, np.nan, dtype = np.float)))
            self._commission = np.concatenate((self._commission, np.full(size, np.nan, dtype = np.float)))
            self._fee = np.concatenate((self._fee, np.full(size, np.nan, dtype = np.float)))
            self._net_pnl = np.concatenate((self._net_pnl, np.full(size, np.nan, dtype = np.float)))
            self._position = np.concatenate((self._position, np.full(size, np.nan, dtype = np.float)))
            self._price = np.concatenate((self._price, np.full(size, np.nan, dtype = np.float)))
            self._open_qty = np.concatenate((self._open_qty, np.full(size, np.nan, dtype = np.float)))
            self._weighted_avg_price = np.concatenate((self._weighted_avg_price, np.full(size, np.nan, dtype = np.float)))
            
            self.size += size
        
    def calc_net_pnl(self, i):
        end_timestamp = self.account_timestamps[i]
        
        if self.contract.expiry is not None and end_timestamp > self.contract.expiry: return
            #raise Exception(f'contract expired: {
        price = self._price_function(self.contract, self.account_timestamps, i, self.strategy_context)
        self._expand_arrays(None, end_timestamp)
        
        i -= self.offset
         
        prev_idx = self.last_net_pnl_calc_idx
        prev_trd_idx = find_last_non_nan_index(self._realized[:i + 1])

        if i < prev_idx: return

        if math.isnan(price):
            self._unrealized[i] = self._unrealized[prev_idx]
        elif math.isclose(self._open_qty[prev_trd_idx], 0):
            self._unrealized[i] = 0
        else:
            # unrealized is simply the difference in ending close and the price * qty of remaining trades
            self._unrealized[i] =  self._open_qty[prev_trd_idx] * (price - self._weighted_avg_price[prev_trd_idx]) * self.multiplier
            
        self._price[i] = price
        
        prev_trd_idx = self.last_trade_calc_idx
        
        if i >= prev_trd_idx:
            self._realized[i] = self._realized[prev_trd_idx]
            self._commission[i] = self._commission[prev_trd_idx]
            self._open_qty[i] = self._open_qty[prev_trd_idx]
            self._weighted_avg_price[i] = self._weighted_avg_price[prev_trd_idx]
            self._fee[i] = self._fee[prev_trd_idx]
            self._position[i] = self._position[prev_trd_idx]
            self.last_net_pnl_calc_idx = i

        self._net_pnl[i] = self._realized[i] + self._unrealized[i] - self._commission[i] - self._fee[i]
        
        if math.isnan(self._net_pnl[i]):
            raise Exception(f'net_pnl: nan i: {i} realized: {self._realized[i]}' + (
                            f' unrealized: {self._unrealized[i]} commission: f{self._commission[i]} fee: {self._fee[i]}'))
        
    def position(self, i):
        if self.contract.expiry is not None and self.account_timestamps[i] > self.contract.expiry:
            return self._position[-1]
        return self._position[i - self.offset]
    
    def net_pnl(self, i):
        if self.contract.expiry is not None and self.account_timestamps[i] > self.contract.expiry:
            return self._net_pnl[-1]
        return self._net_pnl[i - self.offset]
    
    def df(self):
        '''Returns a pandas dataframe with pnl data'''
        df = pd.DataFrame({'timestamp' : self._timestamps, 
                           'unrealized' : self._unrealized,
                           'realized' : self._realized, 
                           'commission' : self._commission,
                           'fee' : self._fee, 
                           'net_pnl' : self._net_pnl,
                           'position' : self._position,
                           'price' : self._price})
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
        first_non_nan_index = np.ravel(np.nonzero(~math.isnan(vals)))
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
        self._trades = []
        
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
            self.symbol_pnls[trade.contract.symbol]._add_trades([trade])
        self._trades += trades
        
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
                symbol_pnl.calc_net_pnl(calc_index)
                net_pnl_diff = symbol_pnl.net_pnl(calc_index) - symbol_pnl.net_pnl(prev_calc_index)
                if math.isfinite(net_pnl_diff): self._equity[calc_index] += net_pnl_diff
                    
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
        return [trade for trade in self._trades if (start_date is None or trade.timestamp >= start_date) and (
            end_date is None or trade.timestamp <= end_date) and (
            contract_group is None or trade.contract.contract_group in contract_groups)]
               
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
        ret_df = pd.merge(ret_df, df_equity, on = ['timestamp'], how = 'left')
        ret_df = ret_df.sort_values(by = ['timestamp', 'contract_group', 'symbol'])
        return ret_df[['timestamp', 'contract_group', 'symbol', 'position', 'price', 'unrealized', 'realized', 
                       'commission', 'fee', 'net_pnl', 'equity']]
    
    def df_trades(self, contract_group = None, start_date = None, end_date = None):
        '''Returns a dataframe of trades
        Args:
            contract_group (:obj:`ContractGroup`, optional): Return trades for this contract group.  
                If None (default), include all contract groups
            start_date (:obj:`np.datetime64`, optional): Include trades with date greater than or equal to this timestamp.
            end_date (:obj:`np.datetime64`, optional): Include trades with date less than or equal to this timestamp.
        '''
        start_date, end_date = str2date(start_date), str2date(end_date)
        trades = self.trades(contract_group, start_date, end_date)
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
    
def test_account():
    #TODO: Fix the tests here
#if __name__ == "__main__":
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
    trade_1 = Trade(ibm_contract, np.datetime64('2018-01-02 08:00'), 10, 10.1, commission = 0.01, 
                    order = MarketOrder(ibm_contract, np.datetime64('2018-01-01 09:00'), 10))
    trade_2 = Trade(ibm_contract, np.datetime64('2018-01-02 09:00'), -20, 15.1, commission = 0.02, 
                    order = MarketOrder(ibm_contract, np.datetime64('2018-01-01 09:00'), -20))
    trade_3 = Trade(msft_contract, timestamps[1], 20, 13.2, commission = 0.04, 
                    order = MarketOrder(msft_contract, timestamps[1], 15))
    trade_4 = Trade(msft_contract, timestamps[2], 20, 16.2, commission = 0.05, 
                    order = MarketOrder(msft_contract, timestamps[2], 20))

    account.add_trades([trade_1, trade_2, trade_3, trade_4])
    # After numpy 1.13 positive floats don't have a leading space for sign
    np.set_printoptions(formatter = {'float' : lambda x : f'{x:.10g}'})
    account.calc(3)
    assert(len(account.df_trades()) == 4)
    assert(len(account.df_pnl()) == 8)
    assert(np.allclose(np.array([1000000, 1000000, 1000183.88, 1000183.88, 1000213.88, 1000213.88]), 
                       account.df_pnl().equity.dropna().values, rtol = 0))
    assert(np.allclose(np.array([0, 10, 20, -10, 40, -10, 40]), account.df_pnl().position.values[1:], rtol = 0))
    assert(np.allclose(np.array([1000000, 1000183.88, 1000213.88]), account.df_pnl([ibm_cg]).equity.dropna().values, rtol = 0))
    
if __name__ == "__main__":
    test_account()
    import doctest
    doctest.testmod(optionflags = doctest.NORMALIZE_WHITESPACE)

