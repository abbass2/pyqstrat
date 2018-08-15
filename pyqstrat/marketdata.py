
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


# In[3]:


def _sort_ohlcv(a):
    l = ['o', 'h', 'l', 'c', 'v']
    if a in l:
        return l.index(a)
    else:
        return -1

class MarketData:
    '''Used to store OHLCV bars.  You must at least supply dates and close prices.  All other fields are optional.
    
    Attributes:
        dates: A numpy datetime array with the datetime for each bar.  Must be monotonically increasing.
        c:     A numpy float array with close prices for the bar.
        o:     A numpy float array with open prices 
        h:     A numpy float array with high prices
        l:     A numpy float array with high prices
        v:     A numpy integer array with volume for the bar
    '''
    def __init__(self, dates, c, o = None, h = None, l = None, v = None):
        '''Zeroes in o, h, l, c are set to nan'''
        assert(len(dates) > 1)
        assert(len(c) == len(dates))
        assert(o is None or len(o) == len(dates))
        assert(h is None or len(h) == len(dates))
        assert(l is None or len(l) == len(dates))
        assert(v is None or len(v) == len(dates))
        
        if not np.all(np.diff(dates).astype(np.float) > 0): # check for monotonically increasing dates
            raise Exception('marketdata dates must be unique monotonically increasing')
            
        self.dates = dates
        self.o = zero_to_nan(o)
        self.h = zero_to_nan(h)
        self.l = zero_to_nan(l)
        self.c = zero_to_nan(c)
        self.v = v
        self._set_valid_rows()
        
    def _set_valid_rows(self):
        nans = np.any(np.isnan([self.o, self.h, self.l, self.c]), axis = 0)
        self.valid_rows = ~nans
    
    def valid_row(self, i):
        '''Return True if the row with index i has no nans in it.'''
        return self.valid_rows[i]
    
    def resample(self, sampling_frequency, inplace = False):
        '''
        Downsample the OHLCV data into a new bar frequency
        
        Args:
            sampling_frequency: See sampling frequency in pandas
            inplace: If set to False, don't modify this object, return a new object instead.
        '''
        if sampling_frequency is None: return self
        df = self.df()
        orig_columns = df.columns
        df = df.resample(sampling_frequency).agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v' : 'sum'}).dropna(how = 'all')
        if not inplace:
            md = MarketData(self.dates, self.c, self.o, self.h, self.l, self.v)
        else:
            md = self
        for col in ['o', 'h', 'l', 'c', 'v']:
            if col in orig_columns: setattr(md, col, df[col].values)
        md.dates = df.index.values
        md._set_valid_rows()
        return md
    
    def errors(self, display = True):
        '''Returns a dataframe indicating any highs that are lower than opens, closes, lows or lows that are higher than other columns
        Also includes any ohlcv values that are negative
        '''
        df = self.df()
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
        for col in ['o', 'h', 'l', 'c', 'v']:
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
        '''Returns a dataframe indicating any values where the bar over bar change is more than warn_std standard deviations.
        
        Args:
            warn_std: Number of standard deviations to use as a threshold (default 10)
            display:  Whether to print out the warning dataframe as well as returning it
        '''
        df = self.df()
        warnings_list = []

        for col in ['o', 'h', 'l', 'c']:
            if col in df.columns:
                data = df[col]
                ret = np.abs(df[col].pct_change())
                std = ret.std()
                mask = ret > warn_std * std
                df_tmp = df[mask]
                if len(df_tmp):
                    double_mask = mask | mask.shift(-1) # Add the previous row so we know the two values computing a return
                    df_tmp = df[double_mask]
                    df_tmp.insert(len(df_tmp.columns), 'ret', ret[mask])
                    df_tmp.insert(len(df_tmp.columns), 'warning', '{} ret > {} std: {}'.format(col, warn_std, round(std, 6)))
                    warnings_list.append(df_tmp)

        if not len(warnings_list): return None
        df = pd.concat(warnings_list)
        if display: dsp.display(df)
        return df
                              
    def overview(self, display = True):
        '''Returns a dataframe showing basic information about the data, including count, number and percent missing, min, max
        
        Args:
            display:  Whether to print out the warning dataframe as well as returning it
        '''
        df = self.df().reset_index()
        df_overview = pd.DataFrame({'count': len(df), 'num_missing' : df.isnull().sum(), 'pct_missing': df.isnull().sum() / len(df), 'min' : df.min(), 'max' : df.max()})
        df_overview = df_overview.T
        columns = sorted(list(df_overview.columns), key = _sort_ohlcv)
        df_overview = df_overview[columns]
        if display: dsp.display(df_overview)
        return df_overview
       
    def time_distribution(self, frequency = '15 minutes', display = True, plot = True, figsize = None):
        '''
        Return a dataframe with the time distribution of the bars
        
        Args:
            frequency: The width of each bin (default "15 minutes").  You can use hours or days as well.
            display:   Whether to display the data in addition to returning it.
            plot:      Whether to plot the data in addition to returning it.
            figsize:   If plot is set, optional figure size for the plot (default (20,8))
        '''
        group_col = None
        
        n = int(frequency.split(' ')[0])
        freq = frequency.split(' ')[1]
        
        df = self.df().reset_index()
        
        if freq == 'minutes' or freq == 'mins' or freq == 'min':
            group_col = [df.date.dt.hour, df.date.dt.minute // n * n]
            names = ['hour', 'minute']
        elif freq == 'hours' or freq == 'hrs' or freq == 'hr':
            group_col = [df.date.dt.weekday_name, df.date.dt.hour // n * n]
            names = ['weekday', 'hour']
        elif freq == 'weekdays' or freq == 'days' or freq == 'day':
            group_col = df.date.dt.weekday_name // n * n
            names = ['weekday']
        else:
            raise Exception(f'unknown time freq: {freq}')
            
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
            if not has_display():
                print('no display found, cannot plot time distribution')
                return tdf
            tdf[cols].plot(figsize = figsize, kind = 'bar', subplots = True, title = 'Time Distribution')
            
        return tdf
    
    def freq_str(self):
        
        freq = infer_frequency(self.dates)
        if freq < 1:
            freq_str = f'{round(freq * 24. * 60, 2)} minutes'
        else:
            freq_str = f'{freq} days'
        return freq_str
            
    def describe(self, warn_std = 10, time_distribution_frequency = '15 min', print_time_distribution = False):
        '''
        Describe the bars.  Shows an overview, errors and warnings for the bar data.  This is a good function to use 
        before running any backtests on a set of bar data.
        
        Args:
            warn_std: See warning function
            time_distribution_frequency: See time_distribution function
            print_time_distribution: Whether to print the time distribution in addition to plotting it.
        '''
        print(f'Inferred Frequency: {self.freq_str()}')
        self.overview()
        print('Errors:')
        self.errors()
        print('Warnings:')
        self.warnings(warn_std = warn_std)
        print('Time distribution:')
        self.time_distribution(display = print_time_distribution, frequency = time_distribution_frequency)
        
    def is_ohlc(self):
        '''
        Returns True if we have all ohlc columns and none are empty
        '''
        return not (self.o is None or self.h is None or self.l is None or self.c is None)

    def plot(self, figsize = (15,8), date_range = None, sampling_frequency = None, title = 'Price / Volume'):
        '''
        Plot a candlestick or line plot depending on whether we have ohlc data or just close prices
        
        Args:
            figsize: Size of the figure (default (15,8))
            date_range: A tuple of strings or numpy datetimes for plotting a smaller sample of the data, e.g. ("2018-01-01", "2018-01-06")
            sampling_frequency: Downsample before plotting.  See pandas frequency strings for possible values.
            title: Title of the graph, default "Price / Volume"
        '''
        date_range = strtup2date(date_range)
        if self.is_ohlc():
            data = OHLC('price', self.dates, self.o, self.h, self.l, self.c, self.v)
        else:
            data = TimeSeries('price', self.dates, self.c)
        subplot = Subplot(data)
        plot = Plot([subplot], figsize = figsize, date_range = date_range, sampling_frequency = sampling_frequency, title = title)
        plot.draw()
                              
    def df(self, start_date = None, end_date = None):
        df = pd.DataFrame({'date' : self.dates, 'c' : self.c}).set_index('date')
        for tup in [('o', self.o), ('h', self.h), ('l', self.l), ('v', self.v)]:
            if tup[1] is not None: df.insert(0, tup[0], tup[1])
        if start_date: df = df[df.index.values >= start_date]
        if end_date: df = df[df.index.values <= end_date]
        return df
    
    
def roll_futures(md, date_func, condition_func, expiries = None, return_full_df = False):
    '''Construct a continuous futures dataframe with one row per datetime given rolling logic
    
    Args:
        md: A dataframe containing the columns 'date', 'series', and any other market data, for example, ohlcv data. Date can contain time for sub-daily bars. 
          The series column must contain a different string name for each futures series, e.g. SEP2018, DEC2018, etc.
        date_func: A function that takes the market data object as an input and returns a numpy array of booleans
          True indicates that the future should be rolled on this date if the condition specified in condition_func is met.
          This function can assume that we have all the columns in the original market data object plus the same columns suffixed with _next for the potential series
          to roll over to.
        condition_func: A function that takes the market data object as input and returns a numpy array of booleans.
          True indicates that we should try to roll the future at that row.
        expiries: An optional dataframe with 2 columns, 'series' and 'expiry'.  This should have one row per future series indicating that future's expiry date.
          If you don't pass this in, the function will assume that the expiry column is present in the original dataframe.
        return_full_df: If set, will return the datframe without removing extra dates so you can use your own logic for rolling, including the _next columns and 
          the roll flag
          
    Returns:
        A pandas DataFrame with one row per date, which contains the columns in the original md DataFrame and the same columns suffixed with _next 
          representing the series we want to roll to.  There is also a column called roll_flag which is set to True whenever 
          the date and roll condition functions are met.
          
    >>> md = pd.DataFrame({'date' : np.concatenate((np.arange(np.datetime64('2018-03-11'), np.datetime64('2018-03-16')),
    ...                                            np.arange(np.datetime64('2018-03-11'), np.datetime64('2018-03-16')))),
    ...                    'c' : [10, 10.1, 10.2, 10.3, 10.4] + [10.35, 10.45, 10.55, 10.65, 10.75],
    ...                    'v' : [200, 200, 150, 100, 100] + [100, 50, 200, 250, 300],
    ...                    'series' : ['MAR2018'] * 5 + ['JUN2018'] * 5})[['date','series', 'c', 'v']]
    >>> expiries = pd.Series(np.array(['2018-03-15', '2018-06-15'], dtype = 'M8[D]'), index = ['MAR2018', 'JUN2018'], name = "expiry")
    >>> date_func = lambda md : md.expiry - md.date <= np.timedelta64(3, 'D')
    >>> condition_func = lambda md : md.v_next > md.v

    >>> df = roll_futures(md, date_func, condition_func, expiries)
    >>> df[df.series == 'MAR2018'].date.max() == np.datetime64('2018-03-14')
    True
    >>> df[df.series == 'JUN2018'].date.max() == np.datetime64('2018-03-15')
    True
    '''
    if 'date' not in md.columns or 'series' not in md.columns:
        raise Exception('date or series not found in columns: {md.columns}')
        
    if expiries is not None:
        expiries = expiries.to_frame(name = 'expiry')
        md = pd.merge(md, expiries, left_on = ['series'], right_index = True, how = 'left')
    else:
        if 'expiry' not in md.columns: raise Exception('expiry column must be present in market data if expiries argument is not specified')
        expiries = md[['series', 'expiry']].drop_duplicates().sort_values(by = 'expiry').set_index('s')

    expiries = pd.merge(expiries, expiries.shift(-1), left_index = True, right_index = True, how = 'left', suffixes = ['', '_next'])

    orig_cols = [col for col in md.columns if col not in ['date']]
    md1 = pd.merge(md, expiries[['expiry', 'expiry_next']], on = ['expiry'], how = 'left')
    md = pd.merge(md1, md, left_on = ['date', 'expiry_next'], right_on = ['date', 'expiry'], how = 'left', suffixes = ['', '_next'])

    md.sort_values(by = ['expiry', 'date'], inplace = True)

    roll_flag = date_func(md) & condition_func(md) 

    df_roll = pd.DataFrame({'series' : md.series, 'date' : md.date, 'roll_flag' : roll_flag})
    df_roll = df_roll[df_roll.roll_flag].groupby('series', as_index = False).first()
    md = pd.merge(md, df_roll, on = ['series', 'date'], how = 'left')
    md.roll_flag = md.roll_flag.fillna(False)
    
    cols = ['date'] + orig_cols + [col + '_next' for col in orig_cols] + ['roll_flag']
    md = md[cols]
    
    if return_full_df: return md
    
    df_list = []
    for series, g in md.groupby('expiry'):
        roll_flag = g.roll_flag
        true_values = roll_flag[roll_flag]
        if len(true_values):
            first_true_index = true_values.index[0]
            roll_flag = roll_flag[first_true_index:]
            false_after_true_values = roll_flag[~roll_flag]
            if len(false_after_true_values):
                first_false_after_true_idx = false_after_true_values.index[0]
                g = g.loc[:first_false_after_true_idx]
        df_list.append(g)

    full_df = pd.concat(df_list)
    full_df = full_df.sort_values(by = ['expiry', 'date']).drop_duplicates(subset=['date'])

    return full_df


def test_marketdata():
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

if __name__ == "__main__":
    test_marketdata()

