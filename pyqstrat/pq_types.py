#cell 0
import pandas as pd
import numpy as np
import types
import datetime

#cell 1
class ContractGroup:
    '''A way to group contracts for figuring out which indicators, rules and signals to apply to a contract and for PNL reporting'''

    _group_names = set()
    
    @staticmethod
    def clear():
        '''
        When running Python interactively you may create a ContractGroup with a given name multiple times because you don't restart Python 
        therefore global variables are not cleared.  This function clears global ContractGroups
        '''
        ContractGroup._group_names = set()
        
    @staticmethod
    def create(name):
        '''
         Args:
            name (str): Name of the group
        '''
        if name in ContractGroup._group_names:
            raise Exception(f'Contract group: {name} already exists')
        ContractGroup._group_names.add(name)
        contract_group = ContractGroup()
        contract_group.name = name
        contract_group.contracts = set()
        contract_group.contracts_by_symbol = {}
        return contract_group
        
    def add_contract(self, contract):
        self.contracts.add(contract)
        self.contracts_by_symbol[contract.symbol] = contract
        
    def get_contract(self, symbol):
        return self.contracts_by_symbol.get(symbol)
        
    def __repr__(self):
        return self.name

class Contract:
    _symbol_names = set()


    '''A contract such as a stock, option or a future that can be traded'''
    @staticmethod
    def create(symbol, contract_group, expiry = None, multiplier = 1., properties = None):
        '''
        Args:
            symbol (str): A unique string reprenting this contract. e.g IBM or ESH9
            contract_group (:obj:`ContractGroup`): We sometimes need to group contracts for calculating PNL, for example, you may have a strategy
                which has 3 legs, a long option, a short option and a future or equity used to hedge delta.  In this case, you will be trading
                different symbols over time as options and futures expire, but you may want to track PNL for each leg using a contract group for each leg.
                So you could create contract groups 'Long Option', 'Short Option' and 'Hedge' and assign contracts to these.
            expiry (obj:`np.datetime64` or :obj:`datetime.datetime`, optional): In the case of a future or option, the date and time when the 
                contract expires.  For equities and other non expiring contracts, set this to None.  Default None.
            multiplier (float, optional): If the market price convention is per unit, and the unit is not the same as contract size, 
                set the multiplier here. For example, for E-mini contracts, each contract is 50 units and the price is per unit, 
                so multiplier would be 50.  Default 1
            properties (obj:`types.SimpleNamespace`, optional): Any data you want to store with this contract.
                For example, you may want to store option strike.  Default None
        '''
        assert(isinstance(symbol, str) and len(symbol) > 0)
        if symbol in Contract._symbol_names:
            raise Exception(f'Contract with symbol: {symbol} already exists')
        Contract._symbol_names.add(symbol)

        #assert(isinstance(contract_group, ContractGroup))
        assert(multiplier > 0)


        contract = Contract()
        contract.symbol = symbol
        
        assert(expiry is None or isinstance(expiry, datetime.datetime) or isinstance(expiry, np.datetime64))
        
        if expiry is not None and isinstance(expiry, datetime.datetime):
            expiry = np.datetime64(expiry)
            
        contract.expiry = expiry
        contract.multiplier = multiplier
        
        if properties is None:
            properties = types.SimpleNamespace()
        contract.properties = properties
        
        contract_group.add_contract(contract)
        contract.contract_group = contract_group
        return contract
    
    @staticmethod
    def clear():
        '''
        When running Python interactively you may create a Contract with a given symbol multiple times because you don't restart Python 
        therefore global variables are not cleared.  This function clears global Contracts
        '''
        Contract._symbol_names = set()
   
    def __repr__(self):
        return f'{self.symbol}' + (f' {self.multiplier}' if self.multiplier != 1 else '') + (
            f' expiry: {self.expiry.astype(datetime.datetime):%Y-%m-%d %H:%M:%S}' if self.expiry is not None else '') + (
            f' group: {self.contract_group.name}' if self.contract_group else '') + (
            f' {self.properties.__dict__}' if self.properties.__dict__ else '')

class Trade:
    def __init__(self, contract, timestamp, qty, price, fee = 0., commission = 0., order = None, properties = None):
        '''
        Args:
            contract (:obj:`Contract`):
            timestamp (:obj:`np.datetime64`): Trade execution datetime
            qty (float): Number of contracts or shares filled
            price (float): Trade price
            fee (float, optional): Fees paid to brokers or others. Default 0
            commision (float, optional): Commission paid to brokers or others. Default 0
            order (:obj:`pq.Order`, optional): A reference to the order that created this trade. Default None
            properties (obj:`types.SimpleNamespace`, optional): Any data you want to store with this contract.
                For example, you may want to store bid / ask prices at time of trade.  Default None
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
        
        if properties is None:
            properties = types.SimpleNamespace()
        self.properties = properties
        
    def __repr__(self):
        '''
        >>> Contract.clear()
        >>> ContractGroup.clear()
        >>> print(Trade(Contract.create('IBM', contract_group = ContractGroup.create('IBM')), np.datetime64('2019-01-01 15:00'), 100, 10.2130000, 0.01))
        IBM 2019-01-01 15:00:00 qty: 100 prc: 10.213 fee: 0.01 order: None
        '''
        timestamp = pd.Timestamp(self.timestamp).to_pydatetime()
        fee = f' fee: {self.fee:.6g}' if self.fee else ''
        commission = f' commission: {self.commission:.6g}' if self.commission else ''
        return f'{self.contract.symbol}' + (
            f' {self.contract.properties.__dict__}' if self.contract.properties.__dict__ else '') + (
            f' {timestamp:%Y-%m-%d %H:%M:%S} qty: {self.qty} prc: {self.price:.6g}{fee}{commission} order: {self.order}') + (
            f' {self.properties.__dict__}' if self.properties.__dict__ else '')
    
class OrderStatus:
    '''
    Enum for order status
    '''
    OPEN = 'open'
    FILLED = 'filled'
    
if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags = doctest.NORMALIZE_WHITESPACE)

