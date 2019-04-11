#cell 0
import pandas as pd
import numpy as np
import types

#cell 1
class ContractGroup:
    '''
    A way to group contracts for figuring out which indicators, rules and signals to apply to a contract and for PNL reporting
    Args:
        name (str): Name of the group
    '''
    def __init__(self, name):
        self.name = name
        self.contracts = set()
        self.contracts_by_symbol = {}
        
    def add_contract(self, contract):
        self.contracts.add(contract)
        self.contracts_by_symbol[contract.symbol] = contract
        
    def get_contract(self, symbol):
        try:
            return self.contracts_by_symbol[symbol]
        except KeyError:
            return None
        
    def __repr__(self):
        return self.name + f' {self.contracts}' if len(self.contracts) else ''

class Contract:
    '''A contract such as a stock, option or a future that can be traded'''
    def __init__(self, symbol, contract_group, multiplier = 1., properties = None):
        '''
        Args:
            symbol (str): A unique string reprenting this contract. e.g IBM or ESH9
            contract_group (:obj:`ContractGroup`): We sometimes need to group contracts for calculating PNL, for example, you may have a strategy
                which has 3 legs, a long option, a short option and a future or equity used to hedge delta.  In this case, you will be trading
                different symbols over time as options and futures expire, but you may want to track PNL for each leg using a contract group for each leg.
                So you could create contract groups 'Long Option', 'Short Option' and 'Hedge' and assign contracts to these.
            multiplier (float, optional): If the market price convention is per unit, and the unit is not the same as contract size, 
                set the multiplier here. For example, for E-mini contracts, each contract is 50 units and the price is per unit, 
                so multiplier would be 50.  Default 1
        '''
        assert(isinstance(symbol, str) and len(symbol) > 0)
        assert(multiplier > 0)
        self.symbol = symbol
        self.multiplier = multiplier
        
        if properties is None:
            properties = types.SimpleNamespace()
        self.properties = properties
        
        contract_group.add_contract(self)
        self.contract_group = contract_group
        
    def __repr__(self):
        return f'{self.symbol} {self.multiplier} ' + (
            f'group: {self.contract_group.name}' if self.contract_group else '') + (
            f' {self.properties}' if self.properties.__dict__ else '')

class Trade:
    def __init__(self, contract, timestamp, qty, price, fee = 0., commission = 0., order = None):
        '''
        Args:
            contract (:obj:`Contract`):
            timestamp (:obj:`np.datetime64`): Trade execution datetime
            qty (float): Number of contracts or shares filled
            price (float): Trade price
            fee (float, optional): Fees paid to brokers or others. Default 0
            commision (float, optional): Commission paid to brokers or others. Default 0
            order (:obj:`pq.Order`, optional): A reference to the order that created this trade. Default None
        '''
        assert(isinstance(contract, Contract))
        assert(np.isfinite(qty))
        assert(np.isfinite(price))
        assert(np.isfinite(fee))
        assert(np.isfinite(commission))
        assert(isinstance(timestamp, np.datetime64))
        
        self.contract = contract
        self.timestamp = timestamp
        self.qty = qty
        self.price = price
        self.fee = fee
        self.commission = commission
        self.order = order
        
    def __repr__(self):
        '''
        >>> print(Trade(Contract('IBM', contract_group = ContractGroup('IBM')), np.datetime64('2019-01-01 15:00'), 100, 10.2130000, 0.01))
        IBM 2019-01-01 15:00:00 qty: 100 prc: 10.213 fee: 0.01 order: None
        '''
        timestamp = pd.Timestamp(self.timestamp).to_pydatetime()
        fee = f' fee: {self.fee:.6g}' if self.fee else ''
        commission = f' commission: {self.commission:.6g}' if self.commission else ''
        return f'{self.contract.symbol}' + (f' {self.contract.properties}' if self.contract.properties.__dict__ else '') + (
            f' {timestamp:%Y-%m-%d %H:%M:%S} qty: {self.qty} prc: {self.price:.6g}{fee}{commission} order: {self.order}')

#cell 2
x = types.SimpleNamespace()

#cell 3
if x.__dict__: print('hello')

#cell 4
if not x: print('hello')

#cell 5


