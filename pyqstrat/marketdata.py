
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import IPython.display as dsp
import matplotlib.dates as mdates
from pyqstrat.pq_utils import *
from pyqstrat.plot import *


# In[28]:


def sort_ohlcv_key(a):
    l = ['date', 'o', 'h', 'l', 'c', 'v', 'vwap']
    if a in l:
        return l.index(a)
    else:
        return len(l)
    
def sort_ohlcv(columns):
    columns = sorted(list(columns)) # Use stable sort to sort columns that we don't know about alphabetically
    return sorted(columns, key = sort_ohlcv_key)

class MarketDataCollection:
    '''
    Used to store a set of market data linking symbol -> MarketData 
    '''
    def __init__(self, symbols = None, marketdata_list = None):
        '''
        Args:
            symbols (list of str, optional): Symbols we want to store market data for.  Default None
            marketdata_list (list of MarketData): Corresponding MarketData object.  Default None
        '''
        if symbols is not None or marketdata_list is not None:
            if symbols is None or marketdata_list is None:
                raise Exception('symbols and marketdata must either both be None or not None')
            if len(symbols) != len(marketdata_list):
                raise Exception('symbols and marketdata_list must contain the same number of elements')
            self.marketdata = dict(zip(symbols, marketdata_list))
        else:
            self.marketdata = {}
        
    def add_marketdata(self, symbol, marketdata):
        self.marketdata[symbol] = marketdata
        
    def add_dates(self, dates):
        for md in self.marketdata.values(): md.add_dates(dates)
            
    def dates(self):
        if len(self.marketdata) == 0:
            return np.array([], dtype = np.datetime64)
        else:
            return list(self.marketdata.values())[0].dates
        
    def items(self):
        return self.marketdata.items()
    
class MarketData:
    '''Used to store OHLCV bars, and any additional time series data you want to use to simulate orders and executions.
        You must at least supply dates and close prices.  All other fields are optional.
    
    Attributes:
        dates: A numpy datetime array with the datetime for each bar.  Must be monotonically increasing.
        c:     A numpy float array with close prices for the bar.
        o:     A numpy float array with open prices . Default None
        h:     A numpy float array with high prices. Default None
        l:     A numpy float array with high prices. Default None
        v:     A numpy integer array with volume for the bar. Default None
        vwap:  A numpy float array with the volume weighted average price for the bar.  Default None
        additional_arrays: A dictionary of name -> numpy array you want to add.  Default None
        resample_funcs: A dictionary of functions for resampling each additional array.  Default None.
        fill_funcs: A dictionary of functions for filling empty rows when we add dates.  Default None.
    '''
    def __init__(self, dates, c, o = None, h = None, l = None, v = None, vwap = None, additional_arrays = None, resample_funcs = None, fill_values = None):
        '''Zeroes in o, h, l, c are set to nan'''
        assert(len(dates) > 1)
        assert(len(c) == len(dates))
        assert(o is None or len(o) == len(dates))
        assert(h is None or len(h) == len(dates))
        assert(l is None or len(l) == len(dates))
        assert(v is None or len(v) == len(dates))
        assert(vwap is None or len(vwap) == len(dates))
        
        additional_arrays = {} if additional_arrays is None else additional_arrays
        
        for k, arr in additional_arrays.items():
            assert(len(arr) == len(dates))
            setattr(self, k, arr)
            
        self.additional_col_names = list(additional_arrays.keys())
        self.resample_funcs = {} if resample_funcs is None else resample_funcs
        self.fill_values = {} if fill_values is None else fill_values
        
        if not np.all(np.diff(dates).astype(np.float) > 0): # check for monotonically increasing dates
            raise Exception('marketdata dates must be unique monotonically increasing')
            
        self.dates = dates
        self.o = zero_to_nan(o)
        self.h = zero_to_nan(h)
        self.l = zero_to_nan(l)
        self.c = zero_to_nan(c)
        self.v = v
        self.vwap = vwap
        self._set_valid_rows()
        
    def add_dates(self, dates):
        '''
        Adds new dates to a market data object.  If fill_values was specified we use that to fill in values for any columns 
        for new dates that are not the same as the old dates.
        
        Args:
            dates (np.array of np.datetime64): New dates to add.  Does not have to be sorted or unique
        
        >>> dates = np.array(['2018-01-05', '2018-01-09', '2018-01-10'], dtype = 'M8[ns]')
        >>> c = np.array([8.1, 8.2, 8.3])
        >>> o = np.array([9, 10, 11])
        >>> additional_arrays = {'x' : np.array([5.1, 5.3, 5.5])}
        >>> fill_values = {'x' : 0}
        >>> md = MarketData(dates, c, o, additional_arrays = additional_arrays, fill_values = fill_values)
        >>> new_dates = np.array(['2018-01-07', '2018-01-09'], dtype = 'M8[ns]')
        >>> md.add_dates(new_dates)
        >>> print(md.dates)
        ['2018-01-05T00:00:00.000000000' '2018-01-07T00:00:00.000000000'
         '2018-01-09T00:00:00.000000000' '2018-01-10T00:00:00.000000000']
        >>> np.set_printoptions(formatter = {'float' : lambda x : f'{x:.4f}'})  # After numpy 1.13 positive floats don't have a leading space for sign
        >>> print(md.o, md.c, md.x)
        [9.0000 nan 10.0000 11.0000] [8.1000 nan 8.2000 8.3000] [5.1000 0.0000 5.3000 5.5000]
        '''
        if dates is None or len(dates) == 0: return
        dates = np.unique(dates)
        new_dates = np.setdiff1d(dates, self.dates, assume_unique = True)
        all_dates = np.concatenate([self.dates, new_dates])
        col_list = ['o', 'h', 'l', 'c', 'vwap'] + self.additional_col_names
        sort_index = all_dates.argsort()
        for col in col_list:
            v = getattr(self, col)
            if v is None: continue
            dtype = getattr(self, col).dtype
            fill_value = self.fill_values[col] if col in self.fill_values else get_empty_np_value(dtype)
            v = np.concatenate([v, np.full(len(new_dates), fill_value, dtype = dtype)])
            v = v[sort_index]
            setattr(self, col, v)
        self.dates = np.sort(all_dates)
        self._set_valid_rows
        
    def _get_fill_value(self, col_name):
        dtype = getattr(self, col_name).dtype
        return get_empty_np_value(dtype)
        
    def _set_valid_rows(self):
        col_list = [col for col in [self.o, self.h, self.l, self.c, self.vwap] if col is not None]
        nans = np.any(np.isnan(col_list), axis = 0)
        self.valid_rows = ~nans
    
    def valid_row(self, i):
        '''Return True if the row with index i has no nans in it.'''
        return self.valid_rows[i]
    
    def get_additional_arrays(self):
        ret = {}
        for key in self.additional_col_names:
            ret[key] = getattr(self, key)
        return ret
    
    def resample(self, sampling_frequency, inplace = False):
        '''
        Downsample the OHLCV data into a new bar frequency
        
        Args:
            sampling_frequency: See sampling frequency in pandas
            inplace: If set to False, don't modify this object, return a new object instead.
        '''
        if sampling_frequency is None:
            if inplace: return None
            return self
        
        df = self.df()
        # Rename index from date to dates since our internal variable is called "dates" but the df() function returns a column "date"
        df.index.name = 'dates'

        df = resample_ohlc(df, sampling_frequency, self.resample_funcs)
              
        if inplace:
            md = self
        else:
            # Create a dummy object, will replace everything (except additional arrays and resample funcs) later
            md = MarketData(self.dates, self.c, self.o, self.h, self.l, self.v, self.vwap, self.get_additional_arrays(), self.resample_funcs)
            
        for col in df.columns: setattr(md, col, df[col].values)
        md._set_valid_rows()
        
        return None if inplace else md
    
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
        for col in ['o', 'h', 'l', 'c', 'v', 'vwap']:
            if col in df.columns:
                neg_values_mask |= (df[col] < 0)
        neg_values = df[neg_values_mask]
        if len(neg_values): 
            neg_values.insert(len(df.columns), 'error', 'negative values')
            errors_list.append(neg_values)
            
        if not len(errors_list): return None
            
        df = pd.concat(errors_list)
        df = df[sort_ohlcv(df.columns)]
        
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

        for col in ['o', 'h', 'l', 'c', 'vwap']:
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
                    df_tmp.insert(len(df_tmp.columns), 'warning', f'{col} ret > {warn_std} * std: {std:.5g}')
                    warnings_list.append(df_tmp)

        if not len(warnings_list): return None
        df = pd.concat(warnings_list)
        df = df[sort_ohlcv(df.columns)]
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
        df_overview = df_overview[sort_ohlcv(df_overview.columns)]
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
            data = OHLC('price', self.dates, self.o, self.h, self.l, self.c, self.v, self.vwap)
        else:
            data = TimeSeries('price', self.dates, self.c)
        subplot = Subplot(data)
        plot = Plot([subplot], figsize = figsize, date_range = date_range, sampling_frequency = sampling_frequency, title = title)
        plot.draw()
                              
    def df(self, start_date = None, end_date = None):
        df = pd.DataFrame({'date' : self.dates, 'c' : self.c}).set_index('date')
        for tup in [('o', self.o), ('h', self.h), ('l', self.l), ('v', self.v), ('vwap', self.vwap)]:
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
    vwap = 0.5 * (l + h)
    c[18] = np.nan
    l[85] = 1000
    md = MarketData(dates, c, o, h, l, v, vwap)
    md.describe()
    md.plot(date_range = ('2018-01-02', '2018-01-02 12:00'))

if __name__ == "__main__":
    test_marketdata()

