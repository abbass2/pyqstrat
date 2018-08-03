
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed") # another bogus warning, see https://github.com/numpy/numpy/pull/432
import pandas as pd
import numpy as np
import IPython.display as dsp
import matplotlib.dates as mdates
from pyqstrat.pq_utils import *
from pyqstrat.plot import *


# In[2]:


def sort_ohlcv(a):
    l = ['o', 'h', 'l', 'c', 'v']
    if a in l:
        return l.index(a)
    else:
        return -1

def roll_futures(md, date_func, condition_func, expiries = None):
    if expiries is not None:
        expiries = expiries.to_frame(name = 'expiry')
        md = pd.merge(md, expiries, left_on = ['series'], right_index = True, how = 'left')
    else:
        expiries = md[['series', 'expiry']].drop_duplicates().sort_values(by = 'expiry').set_index('s')

    expiries = pd.merge(expiries, expiries.shift(-1), left_index = True, right_index = True, how = 'left', suffixes = ['', '_next'])

    md = pd.merge(md, expiries[['expiry', 'expiry_next']], on = ['expiry'], how = 'left')
    md = pd.merge(md, md[['date', 'expiry', 'c']], left_on = ['date', 'expiry_next'], right_on = ['date', 'expiry'], how = 'left', suffixes = ['', '_next'])
    del md['expiry_next']

    md.sort_values(by = ['expiry', 'date'], inplace = True)
    roll_flag = np_shift(date_func(md) & condition_func(md), 1) # need to shift by 1 so order is executed on correct bar
    roll_df = pd.DataFrame({'series' : md.series, 'date' : md.date, 'roll_flag' : roll_flag})
    roll_df = roll_df[roll_df.roll_flag].groupby('series', as_index = False).first()
    md = pd.merge(md, roll_df, on = ['series', 'date'], how = 'left')
    md.roll_flag = md.roll_flag.fillna(False)
    return md

class MarketData:
    def __init__(self, dates, c, o = None, h = None, l = None, v = None):
        if not np.all(np.diff(dates).astype(np.float) > 0): # check for monotonically increasing dates
            raise Exception('marketdata dates must be unique monotonically increasing')
        self.dates = dates
        self.o = zero_to_nan(o)
        self.h = zero_to_nan(h)
        self.l = zero_to_nan(l)
        self.c = zero_to_nan(c)
        self.v = v
        self.set_valid_rows()
        
    def set_valid_rows(self):
        nans = np.any(np.isnan([self.o, self.h, self.l, self.c]), axis = 0)
        self.valid_rows = ~nans
    
    def valid_row(self, i):
        return self.valid_rows[i]
    
    def resample(self, sampling_frequency, inplace = False):
        if sampling_frequency is None: return self
        df = self.to_df()
        orig_columns = df.columns
        df = df.resample(sampling_frequency).agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v' : 'sum'}).dropna(how = 'all')
        if not inplace:
            md = MarketData(self.dates, self.c, self.o, self.h, self.l, self.v)
        else:
            md = self
        for col in ['o', 'h', 'l', 'c']:
            if col in orig_columns: setattr(md, col, df[col].values)
        md.dates = df.index.values
        md.set_valid_rows()
        return md
    
    def errors(self, display = True):
        df = self.to_df()
        errors_list = []
        if 'h' in df.columns:
            bad_highs = df[(df.h < df.c) | (df.h < df.o)]
            if len(bad_highs):                 
                bad_highs.insert(len(df.columns), 'error', 'bad high')
                errors_list.append(bad_highs)
        if 'l' in df.columns:
            bad_lows = df[(df.l > df.c) | (df.l > df.o)]
            if len(bad_lows): 
                bad_lows.insert(len(df.columns), 'error', 'bad low')
                errors_list.append(bad_lows)

        neg_values_mask = (df.c < 0)
        for col in ['o', 'h', 'l', 'v']:
            if col in df.columns:
                neg_values_mask |= (df[col] < 0)
        neg_values = df[neg_values_mask]
        if len(neg_values): 
            neg_values.insert(len(df.columns), 'error', 'negative values')
            errors_list.append(neg_values)
            
        if not len(errors_list): return None
            
        df = pd.concat(errors_list)
        if display: dsp.display(df)
        return df
    
    def warnings(self, warn_std = 10, display = True):
        df = self.to_df()
        warnings_list = []

        for col in ['o', 'h', 'l', 'c']:
            if col in df.columns:
                data = df[col]
                ret = np.abs(df[col].pct_change())
                std = ret.std()
                mask = ret > warn_std * std
                tmp_df = df[mask]
                if len(tmp_df):
                    double_mask = mask | mask.shift(-1) # Add the previous row so we know the two values computing a return
                    tmp_df = df[double_mask]
                    tmp_df.insert(len(tmp_df.columns), 'ret', ret[mask])
                    tmp_df.insert(len(tmp_df.columns), 'warning', f'{col} ret > {warn_std} std: {round(std, 6)}')
                    warnings_list.append(tmp_df)

        if not len(warnings_list): return None
        df = pd.concat(warnings_list)
        if display: dsp.display(df)
        return df
                              
    def overview(self, display = True):
        df = self.to_df().reset_index()
        overview_df = pd.DataFrame({'count': len(df), 'num_missing' : df.isnull().sum(), 'pct_missing': df.isnull().sum() / len(df), 'min' : df.min(), 'max' : df.max()})
        overview_df = overview_df.T
        columns = sorted(list(overview_df.columns), key = sort_ohlcv)
        overview_df = overview_df[columns]
        if display: dsp.display(overview_df)
        return overview_df
        
       
    def time_distribution(self, time_freq = 'minute', display = True, plot = True, figsize = None):
        group_col = None
        
        df = self.to_df().reset_index()
        
        if time_freq == 'minute':
            group_col = [df.date.dt.hour, df.date.dt.minute]
            names = ['hour', 'minute']
        elif time_freq == 'hour':
            group_col = [df.date.dt.weekday_name, df.date.dt.hour]
            names = ['weekday', 'hour']
        elif time_freq == 'weekday':
            group_col = df.date.dt.weekday_name
            names = ['weekday']
        else:
            raise Exception(f'unknown time freq: {time_freq}')
            
        count = df.groupby(group_col)['c'].count()
        tdf = pd.DataFrame({'close_count': count, 'count_pct' : count / df.c.count()})[['close_count', 'count_pct']]
            
        if 'v' in df.columns:
            vsum = df.groupby(group_col)['v'].sum()
            vdf = pd.DataFrame({'volume' : vsum, 'volume_pct' : vsum / df.v.sum()})[['volume', 'volume_pct']]
            tdf = pd.concat([vdf, tdf], axis = 1)
            
        tdf.index.names = names
            
        if display:
            dsp.display(tdf)
    
        if plot:
            if not figsize: figsize = (20, 8)
            cols = ['close_count', 'volume'] if 'v' in df.columns else ['close_count']
            tdf[cols].plot(figsize = figsize, kind = 'bar', subplots = True, title = 'Time Distribution')
            
        return tdf
    
    def freq_str(self):
        freq = infer_frequency(self.dates)
        if freq < 1:
            freq_str = f'{round(freq * 24. * 60, 2)} minutes'
        else:
            freq_str = f'{freq} days'
        return freq_str
            
    def describe(self, warn_std = 10):
        print(f'Inferred Frequency: {self.freq_str()}')
        self.overview()
        print('Errors:')
        self.errors()
        print('Warnings:')
        self.warnings(warn_std = warn_std)
        print('Time distribution:')
        self.time_distribution()
        
    def is_ohlc(self):
        return not (self.o is None or self.h is None or self.l is None or self.c is None)

    def plot(self, figsize = (20,8), date_range = None, sampling_frequency = None, title = 'Price / Volume'):
        date_range = strtup2date(date_range)
        if self.is_ohlc():
            data = OHLC('price', self.dates, self.o, self.h, self.l, self.c, self.v)
        else:
            data = TimeSeries('price', self.dates, self.c)
        subplot = Subplot(data)
        plot = Plot([subplot], figsize = figsize, date_range = date_range, sampling_frequency = sampling_frequency, title = title)
        plot.draw()
                              
    def to_df(self, start_date = None, end_date = None):
        df = pd.DataFrame({'date' : self.dates, 'c' : self.c}).set_index('date')
        for tup in [('o', self.o), ('h', self.h), ('l', self.l), ('v', self.v)]:
            if tup[1] is not None: df.insert(0, tup[0], tup[1])
        if start_date: df = df[df.index.values >= start_date]
        if end_date: df = df[df.index.values <= end_date]
        return df
    
if __name__ == "__main__":
    from datetime import datetime, timedelta
    np.random.seed(0)
    dates = np.arange(datetime(2018, 1, 1, 9, 0, 0), datetime(2018, 3, 1, 16, 0, 0), timedelta(minutes = 5))
    dates = np.array([dt for dt in dates.astype(object) if dt.hour >= 9 and dt.hour <= 16]).astype('M8[m]')
    rets = np.random.normal(size = len(dates)) / 1000
    c_0 = 100
    c = np.round(c_0 * np.cumprod(1 + rets), 2)
    l = np.round(c * (1. - np.abs(np.random.random(size = len(dates)) / 1000.)), 2)
    h = np.round(c * (1. + np.abs(np.random.random(size = len(dates)) / 1000.)), 2)
    o = np.round(l + (h - l) * np.random.random(size = len(dates)), 2)
    v = np.abs(np.round(np.random.normal(size = len(dates)) * 1000))
    c[18] = np.nan
    l[85] = 1000
    md = MarketData(dates, c, o, h, l, v)
    md.describe()
    md.plot(date_range = ('2018-01-02', '2018-01-02 12:00'))

