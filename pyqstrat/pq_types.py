# $$_ Lines starting with # $$_* autogenerated by jup_mini. Do not modify these
# $$_code
# $$_ %%checkall
from __future__ import annotations
import pandas as pd
import numpy as np
import types
import math
import datetime
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, ClassVar
from enum import Enum
from pyqstrat.pq_utils import assert_, get_child_logger

_logger = get_child_logger(__name__)


@dataclass
class ContractGroup:
    '''A way to group contracts for figuring out which indicators, rules and signals to apply to a contract and for PNL reporting'''

    _instances: ClassVar[dict[str, ContractGroup]] = {}
    name: str
    contracts: dict[str, Contract]
        
    def __init__(self, name) -> None:
        self.name = name
        self.contracts = {}
            
    @staticmethod
    def get(name: str) -> ContractGroup:
        '''
        Create a contract group if it does not exist, or returns an existing one
         Args:
            name: Name of the group
        '''
        if name not in ContractGroup._instances:
            cg = ContractGroup(name)
            ContractGroup._instances[name] = cg
            return cg
        return ContractGroup._instances[name]
    
    @staticmethod
    def get_default() -> ContractGroup:
        return DEFAULT_CG
    
    @staticmethod
    def exists(name: str) -> bool:
        return name in ContractGroup._instances
    
    def add_contract(self, contract):
        if contract.symbol not in self.contracts:
            self.contracts[contract.symbol] = contract
        
    def get_contract(self, symbol: str) -> Contract | None:
        return self.contracts.get(symbol)
        
    def get_contracts(self) -> list[Contract]:
        return list(self.contracts.values())
    
    @staticmethod
    def clear_cache() -> None:
        ContractGroup._instances = {}
        
    def clear(self) -> None:
        '''Remove all contracts'''
        self.contracts.clear()
        
    def __repr__(self) -> str:
        return self.name


DEFAULT_CG = ContractGroup.get('DEFAULT')


def _format(obj: SimpleNamespace | None) -> str:
    if obj is None: return ''
    if len(obj.__dict__) == 0: return ''
    return str(obj)


@dataclass
class Contract:
    _instances: ClassVar[dict[str, Contract]] = {}
    symbol: str
    contract_group: ContractGroup
    expiry: np.datetime64 | None
    multiplier: float
    components: list[tuple[Contract, float]]
    properties: SimpleNamespace

    '''A contract such as a stock, option or a future that can be traded'''
    @staticmethod
    def create(symbol: str, 
               contract_group: ContractGroup | None = None, 
               expiry: np.datetime64 | None = None, 
               multiplier: float = 1., 
               components: list[tuple[Contract, float]] | None = None,
               properties: SimpleNamespace | None = None) -> 'Contract':
        '''
        Args:
            symbol: A unique string reprenting this contract. e.g IBM or ESH9
            contract_group: We sometimes need to group contracts for calculating PNL, for example, you may have a strategy
                which has options and a future or equity used to hedge delta.  In this case, you will be trading
                different symbols over time as options and futures expire, but you may want to track PNL for each leg using
                a contract group for each leg. So you could create contract groups 'OPTIONS' and one for 'HEDGES'
            expiry: In the case of a future or option, the date and time when the 
                contract expires.  For equities and other non expiring contracts, set this to None.  Default None.
            multiplier: If the market price convention is per unit, and the unit is not the same as contract size, 
                set the multiplier here. For example, for E-mini contracts, each contract is 50 units and the price is per unit, 
                so multiplier would be 50.  Default 1
            properties: Any data you want to store with this contract.
                For example, you may want to store option strike.  Default None
        '''
        assert_(isinstance(symbol, str) and len(symbol) > 0)
        if contract_group is None: contract_group = DEFAULT_CG
        assert_(symbol not in Contract._instances, f'Contract with symbol: {symbol} already exists')
        assert_(multiplier > 0)
        if components is None: components = []
        if properties is None: properties = types.SimpleNamespace()
        contract = Contract(symbol, contract_group, expiry, multiplier, components, properties)
        contract_group.add_contract(contract)
        contract.contract_group = contract_group
        Contract._instances[symbol] = contract
        return contract
    
    def is_basket(self) -> bool:
        return len(self.components) > 0
    
    @staticmethod
    def exists(name) -> bool:
        return name in Contract._instances
    
    @staticmethod
    def get(name) -> Contract | None:
        '''
        Returns an existing contrat or none if it does not exist
        '''
        return Contract._instances.get(name)
    
    @staticmethod
    def get_or_create(symbol: str, 
                      contract_group: ContractGroup | None = None, 
                      expiry: np.datetime64 | None = None, 
                      multiplier: float = 1., 
                      components: list[tuple[Contract, float]] | None = None,
                      properties: SimpleNamespace | None = None) -> Contract:
        if symbol in Contract._instances:
            contract = Contract._instances.get(symbol)
        else:
            contract = Contract.create(symbol, contract_group, expiry, multiplier, components, properties)
        return contract  # type: ignore
    
    @staticmethod
    def clear_cache() -> None:
        '''
        Remove our cache of contract groups
        '''
        Contract._instances = dict()
        
    def __repr__(self) -> str:
        return f'{self.symbol}' + (f' {self.multiplier}' if self.multiplier != 1 else '') + (
            f' expiry: {self.expiry.astype(datetime.datetime):%Y-%m-%d %H:%M:%S}' if self.expiry is not None else '') + (
            f' group: {self.contract_group.name}' if self.contract_group else '') + (
            f' {_format(self.properties)}')
    

@dataclass
class Price:
    '''
    >>> price = Price(datetime.datetime(2020, 1, 1), 15.25, 15.75, 189, 300)
    >>> print(price)
    15.25@189/15.75@300
    >>> price.properties = SimpleNamespace(delta = -0.3)
    >>> price.valid = False
    >>> print(price)
    15.25@189/15.75@300 delta: -0.3 invalid
    >>> print(price.mid())
    15.5
    '''
    timestamp: datetime.datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    valid: bool = True
    properties: SimpleNamespace | None = None

    @staticmethod
    def invalid() -> Price:
        return Price(datetime.datetime(datetime.MINYEAR, 1, 1),
                     bid=math.nan, 
                     ask=math.nan, 
                     bid_size=-1, 
                     ask_size=-1, 
                     valid=False)
        
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)
    
    def vw_mid(self) -> float:
        '''
        Volume weighted mid
        >>> price = Price(datetime.datetime(2020, 1, 1), 15.25, 15.75, 189, 300)
        >>> print(f'{price.vw_mid():.4f}')
        15.4433
        >>> price.bid_size = 0
        >>> price.ask_size = 0
        >>> assert math.isnan(price.vw_mid())
        '''
        if self.bid_size + self.ask_size == 0: return math.nan
        return (self.bid * self.ask_size + self.ask * self.bid_size) / (self.bid_size + self.ask_size)
    
    def set_property(self, name: str, value: Any) -> None:
        if self.properties is None:
            self.properties = SimpleNamespace()
        setattr(self.properties, name, value)
    
    def spread(self) -> float:
        if self.ask < self.bid: return math.nan
        return self.ask - self.bid
        
    def __repr__(self) -> str:
        msg = f'{self.bid:.2f}@{self.bid_size}/{self.ask:.2f}@{self.ask_size}'
        if self.properties:
            for k, v in self.properties.__dict__.items():
                if isinstance(v, (np.floating, float)):
                    msg += f' {k}: {v:.5g}'
                else:
                    msg += f' {k}: {v}'
        if not self.valid:
            msg += ' invalid'
        return msg


class OrderStatus(Enum):
    '''
    Enum for order status
    '''
    OPEN = 1
    PARTIALLY_FILLED = 2
    FILLED = 3
    CANCEL_REQUESTED = 4
    CANCELLED = 5
    

class TimeInForce(Enum):
    FOK = 1  # Fill or Kill
    GTC = 2  # Good till Cancelled
    DAY = 3  # Cancel at EOD


@dataclass(kw_only=True)
class Order:
    '''
    Args:
        contract: The contract this order is for
        timestamp: Time the order was placed
        qty:  Number of contracts or shares.  Use a negative quantity for sell orders
        reason_code: The reason this order was created. Default ''
        properties: Any order specific data we want to store.  Default None
        status: Status of the order, "open", "filled", etc. Default "open"
    '''
    contract: Contract
    timestamp: np.datetime64 = np.datetime64()
    qty: float = math.nan
    reason_code: str = ''
    time_in_force: TimeInForce = TimeInForce.FOK
    properties: SimpleNamespace = field(default_factory=SimpleNamespace)
    status: OrderStatus = OrderStatus.OPEN
        
    def is_open(self) -> bool:
        return self.status in [OrderStatus.OPEN, OrderStatus.CANCEL_REQUESTED, OrderStatus.PARTIALLY_FILLED]
    
    def request_cancel(self) -> None:
        self.status = OrderStatus.CANCEL_REQUESTED
        
    def fill(self, fill_qty: float = math.nan) -> None:
        assert_(self.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED], 
                f'cannot fill an order in status: {self.status}')
        if math.isnan(fill_qty): fill_qty = self.qty
        assert_(self.qty * fill_qty >= 0, f'order qty: {self.qty} cannot be opposite sign of {fill_qty}')
        assert_(abs(fill_qty) <= abs(self.qty), f'cannot fill qty: {fill_qty} larger than order qty: {self.qty}')
        self.qty -= fill_qty
        if math.isclose(self.qty, 0):
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
        
    def cancel(self) -> None:
        self.status = OrderStatus.CANCELLED
        

@dataclass(kw_only=True)
class MarketOrder(Order):
    def __post_init__(self):
        try:
            if not np.isfinite(self.qty) or math.isclose(self.qty, 0):
                raise ValueError(f'order qty must be finite and nonzero: {self.qty}')
        except Exception as ex:
            _logger.info(ex)
            
    def __repr__(self):
        timestamp = pd.Timestamp(self.timestamp).to_pydatetime()
        return (f'{self.contract.symbol} {timestamp:%Y-%m-%d %H:%M:%S} qty: {self.qty} {self.reason_code}'
                f' {_format(self.properties)} {self.status}')
            

@dataclass(kw_only=True)
class LimitOrder(Order):
    limit_price: float
        
    def __post_init__(self) -> None:
        if not np.isfinite(self.qty) or math.isclose(self.qty, 0):
            raise ValueError(f'order qty must be finite and nonzero: {self.qty}')
            
    def __repr__(self) -> str:
        timestamp = pd.Timestamp(self.timestamp).to_pydatetime()
        symbol = self.contract.symbol if self.contract else ''
        return (f'{symbol} {timestamp:%Y-%m-%d %H:%M:%S} qty: {self.qty} lmt_prc: {self.limit_price}'
                f' {self.reason_code} {_format(self.properties)} {self.status}')


@dataclass(kw_only=True)
class RollOrder(Order):
    close_qty: float
    reopen_qty: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.close_qty) or math.isclose(self.close_qty, 0) \
                or not np.isfinite(self.reopen_qty) or math.isclose(self.reopen_qty, 0):
            raise ValueError(f'order quantities must be non-zero and finite: {self.close_qty} {self.reopen_qty}')
            
    def __repr__(self) -> str:
        timestamp = pd.Timestamp(self.timestamp).to_pydatetime()
        symbol = self.contract.symbol if self.contract else ''
        return (f'{symbol} {timestamp:%Y-%m-%d %H:%M:%S} close_qty: {self.close_qty} reopen_qty: {self.reopen_qty}'
                f' {self.reason_code} {_format(self.properties)} {self.status}')
            

@dataclass(kw_only=True)
class StopLimitOrder(Order):
    '''Used for stop loss or stop limit orders.  The order is triggered when price goes above or below trigger price, depending on whether this is a short
      or long order.  Becomes either a market or limit order at that point, depending on whether you set the limit price or not.
      
    Args:
        trigger_price: Order becomes a market or limit order if price crosses trigger_price.
        limit_price: If not set (default), order becomes a market order when price crosses trigger price.  
            Otherwise it becomes a limit order.  Default np.nan
    '''
    trigger_price: float
    limit_price: float = np.nan
    triggered: bool = False
    
    def __post_init__(self) -> None:
        if not np.isfinite(self.qty) or math.isclose(self.qty, 0):
            raise ValueError(f'order qty must be finite and nonzero: {self.qty}')
    
    def __repr__(self) -> str:
        timestamp = pd.Timestamp(self.timestamp).to_pydatetime()
        symbol = self.contract.symbol if self.contract else ''
        return (f'{symbol} {timestamp:%Y-%m-%d %H:%M:%S} qty: {self.qty} trigger_prc: {self.trigger_price} limit_prc: {self.limit_price}'
                f' {self.reason_code} {_format(self.properties)} {self.status}')
    
    
@dataclass(kw_only=True)
class VWAPOrder(Order):
    '''
    An order type to trade at VWAP. A vwap order executes at VWAP from the point it is sent to the market
    till the vwap end time specified in the order.

    Args:
        vwap_stop: limit price. If market price <= vwap_stop for buys or market price
        >= vwap_stop for sells, the order is executed at that point.
        vwap_end_time: We want to execute at VWAP computed from now to this time
    '''
    vwap_stop: float = math.nan
    vwap_end_time: np.datetime64
        
    def __repr__(self):
        timestamp = pd.Timestamp(self.timestamp).to_pydatetime()
        return (f'{self.contract.symbol} {timestamp:%Y-%m-%d %H:%M:%S} '
                f'limit: {self.vwap_stop:.3f} end: {self.vwap_end_time} qty: {self.qty}'
                f' {self.reason_code} {self.propeties} {self.status}')
            

class Trade:
    def __init__(self, contract: Contract,
                 order: Order,
                 timestamp: np.datetime64, 
                 qty: float, 
                 price: float, 
                 fee: float = 0., 
                 commission: float = 0., 
                 properties: SimpleNamespace | None = None) -> None:
        '''
        Args:
            contract: The contract we traded
            order: A reference to the order that created this trade. Default None
            timestamp: Trade execution datetime
            qty: Number of contracts or shares filled
            price: Trade price
            fee: Fees paid to brokers or others. Default 0
            commision: Commission paid to brokers or others. Default 0
            properties: Any data you want to store with this contract.
                For example, you may want to store bid / ask prices at time of trade.  Default None
        '''
        # assert(isinstance(contract, Contract))
        # assert(isinstance(order, Order))
        assert_(np.isfinite(qty))
        assert_(np.isfinite(price))
        assert_(np.isfinite(fee))
        assert_(np.isfinite(commission))
        # assert(isinstance(timestamp, np.datetime64))
        
        self.contract = contract
        self.order = order
        self.timestamp = timestamp
        self.qty = qty
        self.price = price
        self.fee = fee
        self.commission = commission
        
        if properties is None:
            properties = types.SimpleNamespace()
        self.properties = properties
        
    def __repr__(self) -> str:
        '''
        >>> Contract.clear_cache()
        >>> ContractGroup.clear_cache()
        >>> contract = Contract.create('IBM', contract_group=ContractGroup.get('IBM'))
        >>> order = MarketOrder(contract=contract, timestamp=np.datetime64('2019-01-01T14:59'), qty=100)
        >>> print(Trade(contract, order, np.datetime64('2019-01-01 15:00'), 100, 10.2130000, 0.01))
        IBM 2019-01-01 15:00:00 qty: 100 prc: 10.213 fee: 0.01 order: IBM 2019-01-01 14:59:00 qty: 100 OrderStatus.OPEN
        '''
        timestamp = pd.Timestamp(self.timestamp).to_pydatetime()
        fee = f'fee: {self.fee:.6g}' if self.fee else ''
        commission = f'commission: {self.commission:.6g}' if self.commission else ''
        return (f'{self.contract.symbol} {_format(self.contract.properties)} {timestamp:%Y-%m-%d %H:%M:%S}'
                f' qty: {self.qty} prc: {self.price:.6g} {fee} {commission} order: {self.order} {_format(self.properties)}')
    

@dataclass
class RoundTripTrade:
    contract: Contract
    entry_order: Order
    exit_order: Order | None
    entry_timestamp: np.datetime64
    exit_timestamp: np.datetime64
    qty: int
    entry_price: float
    exit_price: float
    entry_reason: str | None
    exit_reason: str | None
    entry_commission: float
    exit_commission: float
    entry_properties: SimpleNamespace = field(default_factory=SimpleNamespace)
    exit_properties: SimpleNamespace = field(default_factory=SimpleNamespace)
    net_pnl: float = np.nan
    
    
if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
# $$_end_code
