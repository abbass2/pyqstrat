#cell 0
from collections import defaultdict
from sortedcontainers import SortedDict
import math
import pandas as pd
import numpy as np
from copy import copy
from pyqstrat.pq_utils import str2date
from pyqstrat.pq_types import ContractGroup

#cell 1
def calc_trade_pnl(open_qtys, open_prices, new_qtys, new_prices, multiplier):
    '''
    >>> print(calc_trade_pnl(
    ...          open_qtys = np.array([], dtype = np.float), open_prices = np.array([], dtype = np.float), 
    ...            new_qtys = np.array([-8, 9, -4]), new_prices = np.array([10, 11, 6]), multiplier = 100))
    (array([-3.]), array([6.]), -3.0, 6.0, -1300.0)
    >>> print(calc_trade_pnl(open_qtys = np.array([], dtype = np.float), open_prices = np.array([], dtype = np.float), new_qtys = np.array([3, 10, -5]), 
    ...          new_prices = np.array([51, 50, 45]), multiplier = 100))
    (array([8.]), array([50.]), 8.0, 50.0, -2800.0)
    >>> print(calc_trade_pnl(open_qtys = np.array([]), open_prices = np.array([]), 
    ...                new_qtys = np.array([-58, -5, -5, 6, -8, 5, 5, -5, 19, 7, 5, -5, 39]),
    ...                new_prices = np.array([2080, 2075.25, 2070.75, 2076, 2066.75, 2069.25, 2074.75, 2069.75, 2087.25, 2097.25, 2106, 2088.25, 2085.25]),
    ...                multiplier = 50))
    (array([], dtype=float64), array([], dtype=float64), 0.0, 0, -33762.5)    '''
    #TODO: Cythonize this
    
    realized = 0.
    unrealized = 0.
    
    new_qtys = new_qtys.copy()
    new_prices = new_prices.copy()
    

    _open_prices = np.zeros(len(open_prices) + len(new_prices), dtype = np.float)
    _open_prices[:len(open_prices)] = open_prices
    
    _open_qtys = np.zeros(len(open_qtys) + len(new_qtys), dtype = np.float)
    _open_qtys[:len(open_qtys)] = open_qtys
    
    new_qty_indices =  np.nonzero(new_qtys)[0]
    open_qty_indices = np.zeros(len(_open_qtys), dtype = np.int)
    nonzero_indices = np.nonzero(_open_qtys)[0]
    open_qty_indices[:len(nonzero_indices)] = nonzero_indices 

    i = 0                    # index into new_qty_indices to get idx of the new qty we are currently netting
    l = len(nonzero_indices) # virtual length of open_qty_indices
    j = 0                    # index into open_qty_indices to get idx of the open qty we are currently netting
    k = len(open_qtys)       # virtual length of _open_qtys
    
    # Try to net all new trades against existing non-netted trades.
    # Append any remaining non-netted new trades to end of existing trades
    while i < len(new_qty_indices):
        # Always try to net first non-zero new trade against first non-zero existing trade
        # FIFO acccounting
        new_idx = new_qty_indices[i]
        new_qty, new_price = new_qtys[new_idx], new_prices[new_idx]
        
        #print(f'i: {i} j: {j} k: {k} l: {l} oq: {_open_qtys} oqi: {open_qty_indices} op: {_open_prices} nq: {new_qtys} np: {new_prices}')
        
        if j < l: # while we still have open positions to net against
            open_idx = open_qty_indices[j]
            open_qty, open_price = _open_qtys[open_idx], _open_prices[open_idx]
            
            if math.copysign(1, open_qty) == math.copysign(1, new_qty):
                # Nothing to net against so add this trade to the array and wait for the next offsetting trade
                
                _open_qtys[k] = new_qty
                _open_prices[k] = new_price
                open_qty_indices[l] = k
                k += 1
                l += 1

                new_qtys[new_idx] = 0
                i += 1

            elif abs(new_qty) > abs(open_qty):
                # New trade has more qty than offsetting trade so:
                # a. net against offsetting trade
                # b. remove the offsetting trade
                # c. reduce qty of new trade
                open_qty, open_price = _open_qtys[open_idx], _open_prices[open_idx]
                realized += open_qty * (new_price - open_price)
                #print(f'open_qty: {open_qty} open_price: {open_price} open_idx: {open_idx} i: {i} j: {j} k: {k} l: {l} oq: {_open_qtys} oqi: {open_qty_indices} op: {_open_prices} nq: {new_qtys} np: {new_prices}')
                _open_qtys[open_idx] = 0
                j += 1


                new_qtys[new_idx] += open_qty
            else:
                # New trade has less qty than offsetting trade so:
                # a. net against offsetting trade
                # b. remove new trade
                # c. reduce qty of offsetting trade
                realized += new_qty * (open_price - new_price)
                new_qtys[new_idx] = 0
                i += 1
                _open_qtys[open_idx] += new_qty
        else:
            # Nothing to net against so add this trade to the open trades array and wait for the next offsetting trade
            _open_qtys[k] = new_qty
            _open_prices[k] = new_price
            open_qty_indices[l] = k
            k += 1
            l += 1

            new_qtys[new_idx] = 0
            i += 1

    mask = _open_qtys != 0
    _open_qtys = _open_qtys[mask]
    _open_prices = _open_prices[mask]
    open_qty = np.sum(_open_qtys)
    if math.isclose(open_qty, 0):
        weighted_avg_price = 0
    else:
        weighted_avg_price = np.sum(_open_qtys * _open_prices) / open_qty
        
    return _open_qtys, _open_prices, open_qty, weighted_avg_price, realized * multiplier

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

def find_last_non_nan_index(array):
    i = np.nonzero(np.isfinite(array))[0]
    if len(i): return i[-1]
    return 0

def find_index_before(sorted_dict, key):
    '''
    Find index of the first key in a sorted dict that is less than or equal to the key passed in.
    If the key is less than the first key in the dict, return -1
    '''
    size = len(sorted_dict)
    if not size: return -1
    i = sorted_dict.bisect_left(key)
    if i == size: return size - 1
    if sorted_dict.keys()[i] != key:
        return i - 1
    return i

class ContractPNL:
    '''Computes pnl for a single contract over time given trades and market data'''
    def __init__(self, contract, account_timestamps, price_function, strategy_context):
        self.contract = contract
        self._price_function = price_function
        self.strategy_context = strategy_context
        self._account_timestamps = account_timestamps
        self._trade_pnl = SortedDict()
        self._net_pnl = SortedDict()
        # Store trades that are not offset so when new trades come in we can offset against these to calc pnl
        self.open_qtys = np.empty(0, dtype = np.int)
        self.open_prices = np.empty(0, dtype = np.float)
        self.first_trade_timestamp = None
        
    def _add_trades(self, trades):
        '''
        Args:
            trades (list of :obj:`Trade`): Must be sorted by timestamp
        '''
        if not len(trades): return
        timestamps = [trade.timestamp for trade in trades]
        if len(self._trade_pnl):
            k, v = self._trade_pnl.peekitem(0)
            if timestamps[0] <= k:
                raise Exception(f'Can only add a trade that is newer than last added current: {timestamps[0]} prev max timestamp: {k}')
                
        if self.first_trade_timestamp is None: self.first_trade_timestamp = timestamps[0]
                
        for i, timestamp in enumerate(timestamps):
            t_trades = [trade for trade in trades if trade.timestamp == timestamp]
            open_qtys, open_prices, open_qty, weighted_avg_price, realized_chg = calc_trade_pnl(
                self.open_qtys, self.open_prices, 
                np.array([trade.qty for trade in t_trades]), 
                np.array([trade.price for trade in t_trades]),
                self.contract.multiplier)
            self.open_qtys = open_qtys
            self.open_prices = open_prices
            position_chg = sum([trade.qty for trade in t_trades])
            commission_chg = sum([trade.commission for trade in t_trades])
            fee_chg = sum([trade.fee for trade in t_trades])
            index = find_index_before(self._trade_pnl, timestamp)
            if index == -1:
                self._trade_pnl[timestamp] = (position_chg, realized_chg, fee_chg, commission_chg, open_qty, weighted_avg_price)
            else:
                prev_timestamp, (prev_position, prev_realized, prev_fee, prev_commission, _, _) = self._trade_pnl.peekitem(index)
                self._trade_pnl[timestamp] = (prev_position + position_chg, prev_realized + realized_chg, 
                                        prev_fee + fee_chg, prev_commission + commission_chg,  open_qty, weighted_avg_price)
            self.calc_net_pnl(timestamp)
        
    def calc_net_pnl(self, timestamp):
        if timestamp in self._net_pnl: return
        if timestamp < self.first_trade_timestamp: return
        if self.contract.expiry is not None and timestamp > self.contract.expiry: return # We would have calc'ed when the exit trade came in
        i = np.searchsorted(self._account_timestamps, timestamp)
        assert(self._account_timestamps[i] == timestamp)
        
        # Find the index before or equal to current timestamp.  If not found, set to 0's
        trade_pnl_index = find_index_before(self._trade_pnl, timestamp)
        if trade_pnl_index == -1:
            realized, fee, commission, open_qty, open_qty, weighted_avg_price  = 0, 0, 0, 0, 0, 0
        else:
            _, (_, realized, fee, commission, open_qty, weighted_avg_price) = self._trade_pnl.peekitem(trade_pnl_index)
            
        price = np.nan
            
        if not math.isclose(open_qty, 0):
            price = self._price_function(self.contract, self._account_timestamps, i, self.strategy_context)
          
        if math.isnan(price):
            index = find_index_before(self._net_pnl, timestamp) # Last index we computed net pnl for
            if index == -1:
                prev_unrealized = 0
            else:
                _, (_, prev_unrealized, _) = self._net_pnl.peekitem(index)
  
            unrealized = prev_unrealized
        else:
            unrealized = open_qty * (price - weighted_avg_price) * self.contract.multiplier
            
        net_pnl = realized + unrealized - commission - fee

        self._net_pnl[timestamp] = (price, unrealized, net_pnl)
         
        #print(f'{self._trade_pnl} net_pnl: {net_pnl} {self.contract.symbol} i: {i} {timestamp} price: {price} realized: {realized}' + (
        #                        f' unrealized: {unrealized} commission: {commission} fee: {fee} open_qty: {open_qty} wap: {weighted_avg_price}'))
        
    def position(self, timestamp):
        index = find_index_before(self._trade_pnl, timestamp)
        if index == -1: return 0.
        _, (position, _, _, _, _, _) = self._trade_pnl.peekitem(index) # Less than or equal to timestamp
        return position
    
    def net_pnl(self, timestamp):
        index = find_index_before(self._net_pnl, timestamp)
        if index == -1: return 0.
        _, (_, _, net_pnl) = self._net_pnl.peekitem(index) # Less than or equal to timestamp
        return net_pnl
    
    def pnl(self, timestamp):
        index = find_index_before(self._trade_pnl, timestamp)
        position, realized, fee, commission, price, unrealized, net_pnl = 0, 0, 0, 0, 0, 0, 0
        if index != -1:
            _, (position, realized, fee, commission, _, _) = self._trade_pnl.peekitem(index) # Less than or equal to timestamp
        
        index = find_index_before(self._net_pnl, timestamp)
        if index != -1:
            _, (price, unrealized, net_pnl) = self._net_pnl.peekitem(index) # Less than or equal to timestamp
        return position, price, realized, unrealized, fee, commission, net_pnl

    def df(self):
        '''Returns a pandas dataframe with pnl data'''
        df_trade_pnl = pd.DataFrame.from_records([
            (k, v[0], v[1], v[2], v[3]) for k, v in self._trade_pnl.items()],
            columns = ['timestamp', 'position', 'realized', 'fee', 'commission'])
        df_net_pnl = pd.DataFrame.from_records([
            (k, v[0], v[1], v[2]) for k, v in self._net_pnl.items()],
            columns = ['timestamp', 'price', 'unrealized', 'net_pnl'])
        all_timestamps = np.unique(np.concatenate((df_trade_pnl.timestamp.values, df_net_pnl.timestamp.values)))
        #position = df_trade_pnl.set_index('timestamp').position.reindex(all_timestamps, fill_value = 0)
        df_trade_pnl = df_trade_pnl.set_index('timestamp').reindex(all_timestamps, method = 'ffill').reset_index()
        df_trade_pnl = leading_nan_to_zero(df_trade_pnl, ['position', 'realized', 'fee', 'commission'])
        #df_trade_pnl.position = position.values
        df_net_pnl = df_net_pnl.set_index('timestamp').reindex(all_timestamps, method = 'ffill').reset_index()
        del df_net_pnl['timestamp']
        df = pd.concat([df_trade_pnl, df_net_pnl], axis = 1)
        df['symbol'] = self.contract.symbol
        df = df[['symbol', 'timestamp', 'position', 'price', 'unrealized', 'realized', 'commission', 'fee', 'net_pnl']]
        return df
         
def _get_calc_timestamps(timestamps, pnl_calc_time):
    time_delta = np.timedelta64(pnl_calc_time, 'm')
    calc_timestamps = np.unique(timestamps.astype('M8[D]')) + time_delta
    calc_indices = np.searchsorted(timestamps, calc_timestamps, side='left') - 1
    if calc_indices[0] == -1: calc_indices[0] = 0
    return np.unique(timestamps[calc_indices])

class Account:
    '''An Account calculates pnl for a set of contracts'''
    def __init__(self, contract_groups, timestamps, price_function, strategy_context, starting_equity = 1.0e6, pnl_calc_time = 15 * 60):
        '''
        Args:
            contract_groups (list of :obj:`ContractGroup`): Contract groups that we want to compute PNL for
            timestamps (list of np.datetime64): Timestamps that we might compute PNL at
            price_function (function): Function that takes a symbol, timestamps, index, strategy context and 
                returns the price used to compute pnl
            starting_equity (float, optional): Starting equity in account currency.  Default 1.e6
            pnl_calc_time (int, optional): Number of minutes past midnight that we should calculate PNL at.  Default 15 * 60, i.e. 3 pm
        '''
        self.starting_equity = starting_equity
        self._price_function = price_function
        self.strategy_context = strategy_context
        
        self.timestamps = timestamps
        self.calc_timestamps = _get_calc_timestamps(timestamps, pnl_calc_time)
        
        self.contracts = set()
        self._trades = []
        self._pnl = SortedDict()
        
        self.symbol_pnls = {}
        
    def symbols(self):
        return [contract.symbol for contract in self.contracts]
        
    def _add_contract(self, contract, timestamp):
        if contract.symbol in self.symbol_pnls: 
            raise Exception(f'Already have contract with symbol: {contract.symbol} {contract}')
        self.symbol_pnls[contract.symbol] = ContractPNL(contract, self.timestamps, self._price_function, self.strategy_context)
        self.contracts.add(contract)
        
    def add_trades(self, trades):
        trades = sorted(trades, key = lambda x : getattr(x, 'timestamp'))
        # Break up trades by contract so we can add them in a batch
        trades_by_contract = defaultdict(list)
        for trade in trades:
            contract = trade.contract
            if contract not in self.contracts: self._add_contract(contract, trade.timestamp)
            trades_by_contract[contract].append(trade)
            
        for contract, contract_trades in trades_by_contract.items():
            contract_trades.sort(key=lambda x: x.timestamp)
            self.symbol_pnls[contract.symbol]._add_trades(contract_trades)
            
        self._trades += trades
        
    def calc(self, timestamp):
        '''
        Computes P&L and stores it internally for all contracts.
        
        Args:
            timestamp (np.datetime64): timestamp to compute P&L at.  Account remembers the last timestamp it computed P&L up to and will compute P&L
                between these and including timestamp. If there is more than one day between the last index and current index, we will 
                include pnl for at the defined pnl_calc_time for those dates as well.
        '''
        if timestamp in self._pnl: return
            
        prev_idx = find_index_before(self._pnl, timestamp)
        prev_timestamp = None if prev_idx == -1 else self.timestamps[prev_idx]
            
        # Find the last timestamp per day that is between the previous index we computed and the current index,
        # so we can compute daily pnl in addition to the current index pnl
        calc_timestamps = self.calc_timestamps
        intermediate_calc_timestamps = calc_timestamps[calc_timestamps <= timestamp]
        if prev_timestamp is not None:
            intermediate_calc_timestamps = intermediate_calc_timestamps[intermediate_calc_timestamps > prev_timestamp]

        if not len(intermediate_calc_timestamps) or intermediate_calc_timestamps[-1] != timestamp: 
            intermediate_calc_timestamps = np.append(intermediate_calc_timestamps, timestamp)
            
        for ts in intermediate_calc_timestamps:
            net_pnl = 0
            for symbol_pnl in self.symbol_pnls.values():
                symbol_pnl.calc_net_pnl(ts)
                net_pnl += symbol_pnl.net_pnl(ts)
            self._pnl[ts] = net_pnl
        
    def position(self, contract_group, timestamp):
        '''Returns netted position for a contract_group at a given date in number of contracts or shares.'''
        position = 0
        for contract in contract_group.contracts:
            symbol = contract.symbol
            if symbol not in self.symbol_pnls: continue
            position += self.symbol_pnls[symbol].position(timestamp)
        return position
    
    def positions(self, contract_group, timestamp):
        '''
        Returns all non-zero positions in a contract group
        '''
        positions = []
        for contract in contract_group.contracts:
            symbol = contract.symbol
            if symbol not in self.symbol_pnls: continue
            position = self.symbol_pnls[symbol].position(timestamp)
            if not math.isclose(position, 0): positions.append((contract, position))
        return positions
    
    def equity(self, timestamp):
        '''Returns equity in this account in Account currency.  Will cause calculation if Account has not previously 
            calculated up to this date'''
        pnl = self._pnl.get(timestamp)
        if pnl is None:
            self.calc(timestamp)
            pnl = self._pnl[timestamp]
        return self.starting_equity + pnl
    
    def trades(self, contract_group = None, start_date = None, end_date = None):
        '''Returns a list of trades with the given symbol and with trade date between (and including) start date 
            and end date if they are specified. If symbol is None trades for all symbols are returned'''
        start_date, end_date = str2date(start_date), str2date(end_date)
        return [trade for trade in self._trades if (start_date is None or trade.timestamp >= start_date) and (
            end_date is None or trade.timestamp <= end_date) and (
            contract_group is None or trade.contract.contract_group == contract_group)]
               
    def df_pnl(self, contract_groups = None):
        '''
        Returns a dataframe with P&L columns broken down by contract group and symbol
        
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
                if len(df) > 1:
                    net_pnl_diff = np.diff(df.net_pnl.values) # np.diff returns a vector one shorter than the original
                    last_index = np.nonzero(net_pnl_diff)
                    if len(last_index[0]): 
                        last_index = last_index[0][-1] + 1
                        df = df.iloc[:last_index + 1]
                df['contract_group'] = contract_group.name
                dfs.append(df)
        ret_df = pd.concat(dfs)
        ret_df = ret_df.sort_values(by = ['timestamp', 'contract_group', 'symbol'])
        ret_df = ret_df[['timestamp', 'contract_group', 'symbol', 'position', 'price', 'unrealized', 'realized', 
                         'commission', 'fee', 'net_pnl']]
        return ret_df
    
    def df_account_pnl(self, contract_group = None):
        '''
        Returns PNL at the account level.
        
        Args:
            contract_group (:obj:`ContractGroup`, optional): If set, we only return pnl for this contract_group
        '''

        if contract_group is not None:
            symbols = [contract.symbol for contract in contract_group.contracts if contract.symbol in self.symbol_pnls]
            symbol_pnls = [self.symbol_pnls[symbol] for symbol in symbols]
        else:
            symbol_pnls = self.symbol_pnls.values()

        timestamps = self.calc_timestamps
        position = np.full(len(timestamps), 0., dtype = np.float)
        realized = np.full(len(timestamps), 0., dtype = np.float)
        unrealized = np.full(len(timestamps), 0., dtype = np.float)
        fee = np.full(len(timestamps), 0., dtype = np.float)
        commission = np.full(len(timestamps), 0., dtype = np.float)
        net_pnl = np.full(len(timestamps), 0., dtype = np.float)

        for i, timestamp in enumerate(timestamps):
            for symbol_pnl in symbol_pnls:
                _position, _price, _realized, _unrealized, _fee, _commission, _net_pnl = symbol_pnl.pnl(timestamp)
                if math.isfinite(_position): position[i] += _position
                if math.isfinite(_realized): realized[i] += _realized
                if math.isfinite(_unrealized): unrealized[i] += _unrealized
                if math.isfinite(_fee): fee[i] += _fee
                if math.isfinite(_commission): commission[i] += _commission
                if math.isfinite(_net_pnl): net_pnl[i] += _net_pnl

        df = pd.DataFrame.from_records(zip(timestamps, position, unrealized, realized, commission, fee, net_pnl), 
                                       columns = ['timestamp', 'position', 'unrealized', 'realized', 'commission', 'fee', 'net_pnl'])
        df['equity'] = self.starting_equity + df.net_pnl
        return df[['timestamp', 'position', 'unrealized', 'realized', 'commission', 'fee', 'net_pnl', 'equity']]
    
    def df_trades(self, contract_group = None, start_date = None, end_date = None):
        '''
        Returns a dataframe of trades
        
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
                                         (str(trade.order.properties.__dict__) if trade.order.properties.__dict__ else ''), 
                                         (str(trade.contract.properties.__dict__) if trade.contract.properties.__dict__ else '')
                                        ) for trade in trades],
                    columns = ['symbol', 'timestamp', 'qty', 'price', 'fee', 'commission', 'order_date', 'order_qty', 
                               'reason_code', 'order_props', 'contract_props'])
        df = df.sort_values(by = ['timestamp', 'symbol'])
        return df

def test_account():
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
    ContractGroup.clear()
    Contract.clear()
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
    trade_3 = Trade(msft_contract, timestamps[1], 20, 13.2, commission = 0.04, order = MarketOrder(msft_contract, timestamps[1], 15))
    trade_4 = Trade(msft_contract, timestamps[2], 20, 16.2, commission = 0.05, order = MarketOrder(msft_contract, timestamps[2], 20))

    account.add_trades([trade_1, trade_2, trade_3, trade_4])
    account.calc(np.datetime64('2018-01-05 13:35'))
    assert(len(account.df_trades()) == 4)
    assert(len(account.df_pnl()) == 6)
    assert(np.allclose(np.array([9.99,  61.96,  79.97, 103.91,  69.97, 143.91]), 
                       account.df_pnl().net_pnl.values, rtol = 0))
    assert(np.allclose(np.array([10, 20, -10, 40, -10, 40]), account.df_pnl().position.values, rtol = 0))
    
    assert(np.allclose(np.array([1000000.  , 1000183.88, 1000213.88]), account.df_account_pnl().equity.values, rtol = 0))
    
if __name__ == "__main__":
    test_account()
    import doctest
    doctest.testmod(optionflags = doctest.NORMALIZE_WHITESPACE)

