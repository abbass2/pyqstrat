#cell 0
import pandas as pd
import numpy as np

#cell 1
class Trade:
    def __init__(self, symbol, timestamp, qty, price, fee = 0., commission = 0., order = None):
        '''
        Args:
            symbol (str): a string
            timestamp (np.datetime64): Trade execution datetime
            qty (float): Number of contracts or shares filled
            price (float): Trade price
            fee (float): Fees paid to brokers or others. Default 0
            commision (float): Commission paid to brokers or others. Default 0
            order :obj:`pq.Order`: A reference to the order that created this trade. Default None
        '''
        assert(isinstance(symbol, str) and len(symbol) > 0)
        assert(np.isfinite(qty))
        assert(np.isfinite(price))
        assert(np.isfinite(fee))
        assert(np.isfinite(commission))
        assert(isinstance(timestamp, np.datetime64))
        
        self.symbol = symbol
        self.timestamp = timestamp
        self.qty = qty
        self.price = price
        self.fee = fee
        self.commission = commission
        self.order = order
        
    def __repr__(self):
        '''
        >>> print(Trade('IBM', np.datetime64('2019-01-01 15:00'), 100, 10.2130000, 0.01))
        IBM 2019-01-01 15:00 qty: 100 prc: 10.213 fee: 0.01 order: None
        '''
        timestamp = pd.Timestamp(self.timestamp).to_pydatetime()
        fee = f' fee: {self.fee:.6g}' if self.fee else ''
        commission = f' commission: {self.commission:.6g}' if self.commission else ''
        return f'{self.symbol} {timestamp:%Y-%m-%d %H:%M} qty: {self.qty} prc: {self.price:.6g}{fee}{commission} order: {self.order}'
    
class Contract:
    '''A Contract can be a real or virtual instrument. For example, for futures you may wish to create a single continous contract instead of
       a contract for each future series
    '''
    def __init__(self, symbol, multiplier = 1.):
        '''
        Args:
            symbol: A unique string reprenting this contract. e.g IBM or WTI_FUTURE
            multiplier: If you have to multiply price to get price per contract, set that multiplier there.
        '''
        assert(isinstance(symbol, str) and len(symbol) > 0)
        assert(multiplier > 0)
        self.symbol = symbol
        self.multiplier = multiplier
        
    def __repr__(self):
        return f'{self.symbol} multiplier: {self.multiplier}'

#cell 2


