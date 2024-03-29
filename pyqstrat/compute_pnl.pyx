# $$_ Lines starting with # $$_* autogenerated by jup_mini. Do not modify these
# $$_markdown
# # PNL Calculator
# ## Purpose
# cython code for calculating pnl faster
# 

# $$_end_markdown
# $$_code
# $$_ %load_ext cython
# $$_end_code
# $$_code
# $$_ %%cython --force --compile-args=-Wno-parentheses-equality
# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
cimport cython
cimport numpy as np
from libc.stdlib cimport malloc, free
from cython.operator cimport dereference as deref
import numpy as np
import math

cdef long sign(long val):
    return (0 < val) - (val < 0)

cdef struct Trade:
    long qty
    double price

cdef struct TradeVec:
    Py_ssize_t _start_idx
    Py_ssize_t size
    Trade* _trades


cdef void push_back(TradeVec& self, const Trade& trade) except *:
    self.size += 1
    cdef idx = self._start_idx + self.size - 1
    # if idx < 0:
    #    raise RuntimeError(f'invalid idx: {idx}')
    self._trades[idx] = Trade(trade.qty, trade.price)

cdef void pop_front(TradeVec& self) except *:
    self._start_idx += 1
    self.size -= 1
    # if self.size < 0:
    #     raise RuntimeError('size < 0')

cdef Trade* at(const TradeVec& self, int i) except *:
    return &self._trades[self._start_idx + i]

cdef TradeVec create(const long[::1]& qtys, const double[::1]& prices, int extra_capacity) except *:
    cdef Trade* trades = <Trade *>malloc((qtys.shape[0] + extra_capacity) * sizeof(Trade))
    cdef Py_ssize_t i 
    cdef Py_ssize_t size = 0
    for i in range(qtys.shape[0]):
        trades[i] = Trade(qtys[i], prices[i])
        size += 1
    return TradeVec(_start_idx=0, size=size, _trades=trades)

cdef dealloc(TradeVec& self):
    free(self._trades)
    
# cdef print_tv(const TradeVec& tradevec):
#     for i in range(tradevec.size):
#         print(f'{deref(at(tradevec, i)).qty} {deref(at(tradevec, i)).price}')

cdef double net_trade(Trade& trade, Trade& position):
    cdef long abs_tqty
    cdef long abs_pqty
    cdef long txn_qty
    abs_tqty = trade.qty if trade.qty >= 0 else -trade.qty
    abs_pqty = position.qty if position.qty >= 0 else -position.qty
    txn_qty = trade.qty if abs_tqty <= abs_pqty else -position.qty
    cdef double realized = txn_qty * (position.price - trade.price)
    position.qty += txn_qty
    trade.qty -= txn_qty
    return realized

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef calc_trade_pnl(
    long[::1] open_qtys, 
    double[::1] open_prices, 
    long[::1] new_qtys, 
    double[::1] new_prices, 
    double multiplier):
    
    cdef TradeVec trades = create(new_qtys, new_prices, 0)
    cdef TradeVec positions = create(open_qtys, open_prices, new_qtys.shape[0])
            
    cdef Trade* trade
    cdef Trade* position
    cdef long trade_qty
    cdef long pos_qty
    
    cdef double realized = 0
    while True:
        if not trades.size: break
        trade = at(trades, 0)
        trade_qty = deref(trade).qty
             
        if not positions.size:  # no positions, so add the trade
            push_back(positions, deref(trade))
            pop_front(trades)
            continue
            
        position = at(positions, 0)
        pos_qty = deref(position).qty
        
        if sign(trade_qty) == sign(pos_qty):
            push_back(positions, deref(trade))
            pop_front(trades)
            continue
             
        realized += net_trade(deref(position), deref(trade))
        if deref(position).qty == 0:
            pop_front(positions)
        if deref(trade).qty == 0:
            pop_front(trades)
    cdef np.ndarray[long, ndim=1] _open_qtys = np.empty(positions.size, dtype=int)
    cdef np.ndarray[double, ndim=1] _open_prices = np.empty(positions.size, dtype=float)
    for i in range(positions.size):
        _trade = at(positions, i)
        _open_qtys[i] = deref(_trade).qty
        _open_prices[i] = deref(_trade).price
    dealloc(positions)
    dealloc(trades)
    return _open_qtys, _open_prices, realized * multiplier


def calc_trade_pnl_old(open_qtys: np.ndarray, 
                       open_prices: np.ndarray, 
                       new_qtys: np.ndarray, 
                       new_prices: np.ndarray, 
                       multiplier: float) -> tuple[np.ndarray, np.ndarray, float]:
    '''Old slow python implementation 190 us vs 3 us for cython with test case below'''
    realized = 0.
    
    new_qtys = new_qtys.copy()
    new_prices = new_prices.copy()

    _open_prices = np.zeros(len(open_prices) + len(new_prices), dtype=float)
    _open_prices[:len(open_prices)] = open_prices
    
    _open_qtys = np.zeros(len(open_qtys) + len(new_qtys), dtype=float)
    _open_qtys[:len(open_qtys)] = open_qtys
    
    new_qty_indices = np.nonzero(new_qtys)[0]
    open_qty_indices = np.zeros(len(_open_qtys), dtype=int)
    nonzero_indices = np.nonzero(_open_qtys)[0]
    open_qty_indices[:len(nonzero_indices)] = nonzero_indices 

    i = 0                      # index into new_qty_indices to get idx of the new qty we are currently netting
    o = len(nonzero_indices)  # virtual length of open_qty_indices
    j = 0                      # index into open_qty_indices to get idx of the open qty we are currently netting
    k = len(open_qtys)         # virtual length of _open_qtys
    
    # Try to net all new trades against existing non-netted trades.
    # Append any remaining non-netted new trades to end of existing trades
    while i < len(new_qty_indices):
        # Always try to net first non-zero new trade against first non-zero existing trade
        # FIFO acccounting
        new_idx = new_qty_indices[i]
        new_qty, new_price = new_qtys[new_idx], new_prices[new_idx]
        
        # print(f'i: {i} j: {j} k: {k} o: {o} oq: {_open_qtys} oqi: {open_qty_indices} op: {_open_prices} nq: {new_qtys} np: {new_prices}')
        
        if j < o:  # while we still have open positions to net against
            open_idx = open_qty_indices[j]
            open_qty, open_price = _open_qtys[open_idx], _open_prices[open_idx]
            
            if math.copysign(1, open_qty) == math.copysign(1, new_qty):
                # Nothing to net against so add this trade to the array and wait for the next offsetting trade
                
                _open_qtys[k] = new_qty
                _open_prices[k] = new_price
                open_qty_indices[o] = k
                k += 1
                o += 1

                new_qtys[new_idx] = 0
                i += 1

            elif abs(new_qty) > abs(open_qty):
                # New trade has more qty than offsetting trade so:
                # a. net against offsetting trade
                # b. remove the offsetting trade
                # c. reduce qty of new trade
                open_qty, open_price = _open_qtys[open_idx], _open_prices[open_idx]
                realized += open_qty * (new_price - open_price)
                # print(f'open_qty: {open_qty} open_price: {open_price} open_idx: {open_idx} i: {i}
                # j: {j} k: {k} l: {l} oq: {_open_qtys} oqi: {open_qty_indices} op: {_open_prices} nq: {new_qtys} np: {new_prices}')
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
            open_qty_indices[o] = k
            k += 1
            o += 1

            new_qtys[new_idx] = 0
            i += 1

    mask = _open_qtys != 0
    _open_qtys = _open_qtys[mask]
    _open_prices = _open_prices[mask]
       
    return _open_qtys, _open_prices, realized * multiplier

# $$_end_code
# $$_code
def test_calc_trade_pnl():
    print('called')
    pos = [(15, 6.), (8, 5.)]
    trades = [[(10, 8.), (-5, 7.)],
              [(-10, 8.), (-5, 7.)],
              [(-20, 8.), (-5, 7.)]]
    
    results = [([10,  8, 10], [6., 5., 8.], 500.0),
               ([8], [5.], 2500.0),
               ([-2], [7.], 5100.0)]
    
    for i, _trades in enumerate(trades):
        open_qtys = np.array(list(zip(*pos))[0])
        open_prices = np.array(list(zip(*pos))[1])
        new_qtys = np.array(list(zip(*_trades))[0])
        new_prices = np.array(list(zip(*_trades))[1])
        # print(f'positions: {open_qtys} {open_prices} trades: {new_qtys} {new_prices}')
        out = calc_trade_pnl(open_qtys, open_prices, new_qtys, new_prices, 100.)
        assert(np.all(out[0] == results[i][0]))
        assert(np.allclose(out[1], results[i][1]))
        assert(np.isclose(out[2], results[i][2]))

        
    trades = [[(-8, 10.), (9, 11.), (-4, 6.)],
              [(3, 51.), (10, 50.), (-5, 45.)],
              list(zip(
                  np.array([-58, -5, -5, 6, -8, 5, 5, -5, 19, 7, 5, -5, 39], dtype=int), 
                  np.array([2080, 2075.25, 2070.75, 2076, 2066.75, 2069.25, 2074.75, 
                            2069.75, 2087.25, 2097.25, 2106, 2088.25, 2085.25], dtype=float)))]
    results = [([-3], [6.], -1300.0),
               ([8], [50.], -2800.0),
               ([], [], -67525.0)]
                 
    
    for i, _trades in enumerate(trades):
        open_qtys = np.empty(0, dtype=int)
        open_prices = np.empty(0, dtype=float)
        new_qtys = np.array(list(zip(*_trades))[0])
        new_prices = np.array(list(zip(*_trades))[1])
        # print(f'positions: {open_qtys} {open_prices} trades: {new_qtys} {new_prices}')
        out = calc_trade_pnl(open_qtys, open_prices, new_qtys, new_prices, 100.)
        assert(np.all(out[0] == results[i][0]))
        assert(np.allclose(out[1], results[i][1]))
        assert(np.isclose(out[2], results[i][2]))
        
        
if __name__ == '__main__':
    np.random.seed(0)
    pos_size = 100
    open_qtys = np.random.randint(1, 100, pos_size)
    open_prices = np.random.normal(10., 1, pos_size)
    trade_size = 90
    new_qtys = np.random.randint(-20, 20, trade_size)
    new_prices = np.random.normal(10., 1, trade_size)
# $$_     %timeit calc_trade_pnl(open_qtys, open_prices, new_qtys, new_prices, 100)
    test_calc_trade_pnl()
# $$_end_code
