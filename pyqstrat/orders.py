#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import types
import numpy as np
import pandas as pd
from pyqstrat.pq_utils import ReasonCode

class MarketOrder:
    def __init__(self, contract, timestamp, qty, reason_code = ReasonCode.NONE, properties = None, status = 'open'):
        '''
        Args:
            contract (:obj:`Contract`):
            timestamp (:obj:`np.datetime64`): Time the order was placed
            qty (float):  Number of contracts or shares.  Use a negative quantity for sell orders
            reason_code (str, optional): The reason this order was created.
                Prefer a predefined constant from the ReasonCode class if it matches your reason for creating this order.
                Default None
            properties (:obj:`SimpleNamespace`, optional): Any order specific data we want to store.  Default None
            status (str, optional): Status of the order, "open", "filled", etc. (default "open")
        '''
        self.contract = contract
        self.timestamp = timestamp
        if not np.isfinite(qty) or math.isclose(qty, 0): raise Exception(f'order qty must be finite and nonzero: {qty}')
        self.qty = qty
        self.reason_code = reason_code
        self.status = status
        if properties is None: properties = types.SimpleNamespace()
        self.properties = properties
        
    def __repr__(self):
        timestamp = pd.Timestamp(self.timestamp).to_pydatetime()
        return f'{self.contract.symbol} {timestamp:%Y-%m-%d %H:%M:%S} qty: {self.qty}' + (
            '' if self.reason_code == ReasonCode.NONE else f' {self.reason_code}') + (
            '' if not self.properties.__dict__ else f' {self.properties}') + (
            f' {self.status}')
            
class LimitOrder:
    def __init__(self, contract, timestamp, qty, limit_price, reason_code = ReasonCode.NONE, properties = None, status = 'open'):
        '''
        Args:
            contract (:obj:`Contract`):
            timestamp (:obj:`np.datetime64`): Time the order was placed
            qty (float):  Number of contracts or shares.  Use a negative quantity for sell orders
            limit_price (float): Limit price (float)
            reason_code (str, optional): The reason this order was created.
                Prefer a predefined constant from the ReasonCode class if it matches your reason for creating this order.
                Default None
            properties (:obj:`SimpleNamespace`, optional): Any order specific data we want to store.  Default None
            status (str, optional): Status of the order, "open", "filled", etc. (default "open")
        '''
        self.contract = contract
        self.timestamp = timestamp
        if not np.isfinite(qty) or math.isclose(qty, 0): raise Exception(f'order qty must be finite and nonzero: {qty}')
        self.qty = qty
        self.reason_code = reason_code
        self.limit_price = limit_price
        if properties is None: properties = types.SimpleNamespace()
        self.properties = properties
        self.properties.limit_price = self.limit_price
        self.status = status
        
    def __repr__(self):
        timestamp = pd.Timestamp(self.timestamp).to_pydatetime()
        return f'{self.contract.symbol} {timestamp:%Y-%m-%d %H:%M:%S} qty: {self.qty} lmt_prc: {self.limit_price}' + (
            '' if self.reason_code == ReasonCode.NONE else f' {self.reason_code}') + (
            '' if not self.properties.__dict__ else f' {self.properties}') + (
            f' {self.status}')
    
    
class RollOrder:
    '''A roll order is used to roll a future from one series to the next.  It represents a sell of one future and the buying of another future.'''
    def __init__(self, contract, timestamp, close_qty, reopen_qty, reason_code = ReasonCode.ROLL_FUTURE, properties = None, 
                 status = 'open'):
        '''
        Args:
            contract (:obj:`Contract`):
            timestamp (:obj:`np.datetime64`): Time the order was placed
            close_qty (float): Quantity of the future you are rolling
            reopen_qty (float): Quantity of the future you are rolling to
            reason_code (str, optional): The reason this order was created.
                Prefer a predefined constant from the ReasonCode class if it matches your reason for creating this order.
                Default None
            properties (:obj:`SimpleNamespace`, optional): Any order specific data we want to store.  Default None
            status (str, optional): Status of the order, "open", "filled", etc. (default "open")
        '''
        self.contract = contract
        self.timestamp = timestamp
        if not np.isfinite(close_qty) or math.isclose(close_qty, 0) or not np.isfinite(reopen_qty) or math.isclose(reopen_qty, 0):
            raise Exception(f'order quantities must be non-zero and finite: {close_qty} {reopen_qty}')
        self.close_qty = close_qty
        self.reopen_qty = reopen_qty
        self.reason_code = reason_code
        self.qty = close_qty # For display purposes when we print varying order types
        if properties is None: properties = types.SimpleNamespace()
        self.properties = properties
        self.properties.close_qty = self.close_qty
        self.properties.reopen_qty = self.reopen_qty
        self.status = status
        
    def __repr__(self):
        timestamp = pd.Timestamp(self.timestamp).to_pydatetime()
        return f'{self.contract.symbol} {timestamp:%Y-%m-%d %H:%M:%S} close_qty: {self.close_qty} reopen_qty: {self.reopen_qty}' + (
            '' if self.reason_code == ReasonCode.NONE else f' {self.reason_code}') + '' if not self.properties.__dict__ else f' {self.properties}' + (
            f' {self.status}')

  
class StopLimitOrder:
    '''Used for stop loss or stop limit orders.  The order is triggered when price goes above or below trigger price, depending on whether this is a short
      or long order.  Becomes either a market or limit order at that point, depending on whether you set the limit price or not.
    '''
    def __init__(self, contract, timestamp, qty, trigger_price, limit_price = np.nan, reason_code = ReasonCode.NONE, 
                 properties = None, status = 'open'):
        '''
        Args:
            contract (:obj:`Contract`):
            timestamp (:obj:`np.datetime64`): Time the order was placed
            qty (float):  Number of contracts or shares.  Use a negative quantity for sell orders
            trigger_price (float): Order becomes a market or limit order if price crosses trigger_price.
            limit_price (float, optional): If not set (default), order becomes a market order when price crosses trigger price.  
                Otherwise it becomes a limit order.  Default np.nan
            reason_code (str, optional): The reason this order was created.
                Prefer a predefined constant from the ReasonCode class if it matches your reason for creating this order.
                Default None
            properties (:obj:`SimpleNamespace`, optional): Any order specific data we want to store.  Default None
            status (str, optional): Status of the order, "open", "filled", etc. (default "open")
        '''      
        self.contract = contract
        self.timestamp = timestamp
        if not np.isfinite(qty) or math.isclose(qty, 0): raise Exception(f'order qty must be finite and nonzero: {qty}')
        self.qty = qty
        self.trigger_price = trigger_price
        self.limit_price = limit_price
        self.reason_code = reason_code
        self.triggered = False
        if properties is None: properties = types.SimpleNamespace()
        self.properties = properties
        self.properties.trigger_price = trigger_price
        self.properties.limit_price = limit_price
        self.status =  status
        
    def __repr__(self):
        timestamp = pd.Timestamp(self.timestamp).to_pydatetime()
        return f'{self.contract.symbol} {timestamp:%Y-%m-%d %H:%M:%S} qty: {self.qty} trigger_prc: {self.trigger_price} limit_prc: {self.limit_price}' + (
            '' if self.reason_code == ReasonCode.NONE else f' {self.reason_code}') + '' if not self.properties.__dict__ else f' {self.properties}'+ (
            f' {self.status}')
    
if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags = doctest.NORMALIZE_WHITESPACE)

