#cell 1
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed") # ignore pesky warning, see https://github.com/numpy/numpy/pull/432

import pandas as pd
import numpy as np
import pyqstrat as pq
from pyqstrat.examples.build_example_strategy import build_example_strategy
from pyqstrat.evaluator import compute_sharpe, compute_sortino, compute_maxdd_pct, compute_amean, compute_rolling_dd



#cell 3
def generator():
    for lookback_period in np.arange(5, 10): 
        for num_std in np.arange(1, 3, 0.5): # number of standard deviations the bands are away from the SMA
            costs = (yield {'lookback_period' : lookback_period, 'num_std' : num_std})
            yield
            
def cost_func(suggestion):
    lookback_period, num_std = suggestion['lookback_period'], suggestion['num_std']
    
    strategy = build_example_strategy(lookback_period = lookback_period, num_std = num_std)
    
    portfolio = pq.Portfolio()
    portfolio.add_strategy('bb_strategy', strategy)
    
    portfolio.run()
    
    returns_df = strategy.df_returns()
    
    returns = returns_df.ret.values
    equity = returns_df.equity.values
    dates = returns_df.index.values
    
    amean = compute_amean(returns)
    sharpe = compute_sharpe(returns, amean, 252)
    sortino = compute_sortino(returns, amean, 252)
    rolling_dd = compute_rolling_dd(dates, equity)
    maxdd = compute_maxdd_pct(rolling_dd)
    
    return sharpe, {'sortino' : sortino, 'maxdd' : maxdd}

optimizer = pq.Optimizer('example', generator(), cost_func, max_processes = 1)
optimizer.run(raise_on_error = True)

optimizer.plot_3d(x = 'lookback_period', y = 'num_std', plot_type = 'surface');
    

#cell 5
optimizer.plot_3d(x = 'lookback_period', y = 'num_std', plot_type = 'contour', hspace = 0.25);

#cell 7
optimizer.df_experiments(ascending = False)

#cell 8
optimizer.df_experiments(sort_column = 'maxdd', ascending = False)

