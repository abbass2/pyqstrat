# type: ignore
# flake8: noqa

import pandas as pd
import numpy as np
import math
from types import SimpleNamespace
import pyqstrat as pq

def sma(contract_group, timestamps, indicators, strategy_context): # simple moving average
    sma = pd.Series(indicators.c).rolling(window = strategy_context.lookback_period).mean()
    return sma.values

def band(contract_group, timestamps, indicators, strategy_context, upper):
    std = pd.Series(indicators.c).rolling(window = strategy_context.lookback_period).std()
    return indicators.sma + strategy_context.num_std * std * (1 if upper else -1)

upper_band = lambda contract_group, timestamps, indicators, strategy_context : \
    band(contract_group, timestamps, indicators, strategy_context, upper = True)

lower_band = lambda contract_group, timestamps, indicators, strategy_context : \
    band(contract_group, timestamps, indicators, strategy_context, upper = False)

def bollinger_band_signal(contract_group, timestamps, indicators, parent_signals, strategy_context):
    # Replace nans with 0 so we don't get errors later when comparing nans to floats
    h = np.nan_to_num(indicators.h)
    l = np.nan_to_num(indicators.l)
    
    upper_band = np.nan_to_num(indicators.upper_band)
    lower_band = np.nan_to_num(indicators.lower_band)
    sma = np.nan_to_num(indicators.sma)
    
    signal = np.where(h > upper_band, 2, 0)
    signal = np.where(l < lower_band, -2, signal)
    signal = np.where((h > sma) & (signal == 0), 1, signal) # price crossed above simple moving avg but not above upper band
    signal = np.where((l < sma) & (signal == 0), -1, signal) # price crossed below simple moving avg but not below lower band
    return signal

def bollinger_band_trading_rule(contract_group, i, timestamps, indicators, signal, account, strategy_context):
    timestamp = timestamps[i]
    curr_pos = account.position(contract_group, timestamp)
    signal_value = signal[i]
    risk_percent = 0.1
    close_price = indicators.c[i]
    
    contract = contract_group.get_contract('PEP')
    if contract is None:
        contract = pq.Contract.create(symbol = 'PEP', contract_group = contract_group)
    
    # if we don't already have a position, check if we should enter a trade
    if math.isclose(curr_pos, 0):
        if signal_value == 2 or signal_value == -2:
            curr_equity = account.equity(timestamp)
            order_qty = np.round(curr_equity * risk_percent / close_price * np.sign(signal_value))
            trigger_price = close_price
            reason_code = pq.ReasonCode.ENTER_LONG if order_qty > 0 else pq.ReasonCode.ENTER_SHORT
            return [pq.StopLimitOrder(contract, timestamp, order_qty, trigger_price, reason_code = reason_code)]
        
    else: # We have a current position, so check if we should exit
        if (curr_pos > 0 and signal_value == -1) or (curr_pos < 0 and signal_value == 1):
            order_qty = -curr_pos
            reason_code = pq.ReasonCode.EXIT_LONG if order_qty < 0 else pq.ReasonCode.EXIT_SHORT
            return [pq.MarketOrder(contract, timestamp, order_qty, reason_code = reason_code)]
    return []

def market_simulator(orders, i, timestamps, indicators, signals, strategy_context):
    trades = []
    timestamp = timestamps[i]
    
    
    for order in orders:
        trade_price = np.nan
        
        cgroup = order.contract.contract_group
        ind = indicators[cgroup]
        
        o, h, l, c = ind.o[i], ind.h[i], ind.l[i], ind.c[i]
        
        if isinstance(order, pq.MarketOrder):
            trade_price = 0.5 * (o + h) if order.qty > 0 else 0.5 * (o + l)
        elif isinstance(order, pq.StopLimitOrder):
            if (order.qty > 0 and h > order.trigger_price) or (order.qty < 0 and l < order.trigger_price): # A stop order
                trade_price = 0.5 * (order.trigger_price + h) if order.qty > 0 else 0.5 * (order.trigger_price + l)
        else:
            raise Exception(f'unexpected order type: {order}')
            
        if np.isnan(trade_price): continue
            
        trade = pq.Trade(order.contract, order, timestamp, order.qty, trade_price, commission = order.qty * 5, fee = 0)
        order.status = 'filled'
                           
        trades.append(trade)
                           
    return trades

def get_price(symbol, timestamps, i, strategy_context):
    return strategy_context.c[i]

def build_example_strategy(strategy_context):

    try:
        file_path = os.path.dirname(os.path.realpath(__file__)) + '/../notebooks/support/pepsi_15_min_prices.csv.gz' # If we are running from unit tests
    except:
        file_path = '../notebooks/support/pepsi_15_min_prices.csv.gz'
    
    prices = pd.read_csv(file_path)
    prices.date = pd.to_datetime(prices.date)

    timestamps = prices.date.values

    pq.ContractGroup.clear()
    pq.Contract.clear()

    contract_group = pq.ContractGroup.create('PEP')

    strategy_context.c = prices.c.values # For use in the get_price function

    strategy = pq.Strategy(timestamps, [contract_group], get_price, trade_lag = 1, strategy_context = strategy_context)
    
    strategy.add_indicator('o', prices.o.values)
    strategy.add_indicator('h', prices.h.values)
    strategy.add_indicator('l', prices.l.values)
    strategy.add_indicator('c', prices.c.values)

    strategy.add_indicator('sma', sma, depends_on = ['c'])
    strategy.add_indicator('upper_band', upper_band, depends_on = ['c', 'sma'])
    strategy.add_indicator('lower_band', lower_band, depends_on = ['c', 'sma'])
    
    strategy.add_signal('bb_signal', bollinger_band_signal, depends_on_indicators = ['h', 'l', 'sma', 'upper_band', 'lower_band'])

    # ask pqstrat to call our trading rule when the signal has one of the values [-2, -1, 1, 2]
    strategy.add_rule('bb_trading_rule', bollinger_band_trading_rule, 
                      signal_name = 'bb_signal', sig_true_values = [-2, -1, 1, 2])

    strategy.add_market_sim(market_simulator)

    return strategy
