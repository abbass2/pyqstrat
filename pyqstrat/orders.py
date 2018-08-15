
# coding: utf-8

# In[1]:


import math
from pyqstrat.pq_utils import *

class MarketOrder:
    def __init__(self, symbol, date, qty, reason_code = ReasonCode.NONE, status = 'open'):
        '''
        Args:
            symbol: A string
            date: A numpy datetime indicating the time the order was placed
            qty:  Number of contracts or shares.  Use a negative quantity for sell orders
            reason_code: A string representing the reason this order was created.
                Prefer a predefined constant from the ReasonCode class if it matches your reason for creating this order.
            status: Status of the order, "open", "filled", etc. (default "open")
        '''
        self.symbol = symbol
        self.date = date
        if not np.isfinite(qty) or math.isclose(qty, 0): raise Exception(f'order qty must be finite and nonzero: {qty}')
        self.qty = qty
        self.reason_code = reason_code
        self.status = status
        
    def __repr__(self):
        return f'{self.symbol} {pd.Timestamp(self.date).to_pydatetime():%Y-%m-%d %H:%M} qty: {self.qty}' + (
            '' if self.reason_code == ReasonCode.NONE else f' {self.reason_code}') + f' {self.status}'
    
    def params(self):
        return {}
        
class LimitOrder:
    def __init__(self, symbol, date, qty, limit_price, reason_code = ReasonCode.NONE, status = 'open'):
        '''
        Args:
            symbol: A string
            date: A numpy datetime indicating the time the order was placed
            qty:  Number of contracts or shares.  Use a negative quantity for sell orders
            limit_price: Limit price (float)
            reason_code: A string representing the reason this order was created.
                Prefer a predefined constant from the ReasonCode class if it matches your reason for creating this order.
            status: Status of the order, "open", "filled", etc. (default "open")
        '''
        self.symbol = symbol
        self.date = date
        if not np.isfinite(qty) or math.isclose(qty, 0): raise Exception(f'order qty must be finite and nonzero: {qty}')
        self.qty = qty
        self.reason_code = reason_code
        self.limit_price = limit_price
        self.status = status
        
    def __repr__(self):
        return f'{self.symbol} {pd.Timestamp(self.date).to_pydatetime():%Y-%m-%d %H:%M} qty: {self.qty} lmt_prc: {self.limit_price}' + (
            '' if self.reason_code == ReasonCode.NONE else f' {self.reason_code}') + f' {self.status}'
    
    def params(self):
        return {'limit_price' : self.limit_price}
    
class RollOrder:
    '''A roll order is used to roll a future from one series to the next.  It represents a sell of one future and the buying of another future.'''
    def __init__(self, symbol, date, close_qty, reopen_qty, reason_code = ReasonCode.ROLL_FUTURE, status = 'open'):
        '''
        Args:
            symbol: A string
            date: A numpy datetime indicating the time the order was placed
            close_qty: Quantity of the future you are rolling
            reopen_qty: Quantity of the future you are rolling to
            reason_code: A string representing the reason this order was created.
                Prefer a predefined constant from the ReasonCode class if it matches your reason for creating this order.
            status: Status of the order, "open", "filled", etc. (default "open")
        '''
        self.symbol = symbol
        self.date = date
        if not np.isfinite(close_qty) or math.isclose(close_qty, 0) or not np.isfinite(reopen_qty) or math.isclose(reopen_qty, 0):
            raise Exception(f'order quantities must be non-zero and finite: {close_qty} {reopen_qty}')
        self.close_qty = close_qty
        self.reopen_qty = reopen_qty
        self.reason_code = reason_code
        self.qty = close_qty # For display purposes when we print varying order types
        self.status = status
        
    def params(self):
        return {'close_qty' : self.close_qty, 'reopen_qty' : self.reopen_qty}
        
    def __repr__(self):
        return f'{self.symbol} {pd.Timestamp(self.date).to_pydatetime():%Y-%m-%d %H:%M} close_qty: {self.close_qty} reopen_qty: {self.reopen_qty}' + (
            '' if self.reason_code == ReasonCode.NONE else f' {self.reason_code}') + f' {self.status}'
  
class StopLimitOrder:
    '''Used for stop loss or stop limit orders.  The order is triggered when price goes above or below trigger price, depending on whether this is a short
      or long order.  Becomes either a market or limit order at that point, depending on whether you set the limit price or not.
    '''
    def __init__(self, symbol, date, qty, trigger_price, limit_price = np.nan, reason_code = ReasonCode.NONE, status = 'open'):
        '''
        Args:
            symbol: A string
            date: A numpy datetime indicating the time the order was placed
            qty: Number of contracts or shares.  Use a negative value for sell orders
            trigger_price: Order becomes a market or limit order if price crosses trigger_price.
            limit_price: If not set (default), order becomes a market order when price crosses trigger price.  Otherwise it becomes a limit order
            reason_code: A string representing the reason this order was created.
                Prefer a predefined constant from the ReasonCode class if it matches your reason for creating this order.
            status: Status of the order, "open", "filled", etc. (default "open")
        '''      
        self.symbol = symbol
        self.date = date
        if not np.isfinite(qty) or math.isclose(qty, 0): raise Exception(f'order qty must be finite and nonzero: {qty}')
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
            '' if self.reason_code == ReasonCode.NONE else f' {self.reason_code}') + f' {self.status}'
                

