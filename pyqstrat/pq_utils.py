
# coding: utf-8

# In[1]:


import os
import numpy as np
import datetime
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

SEC_PER_DAY = 3600 * 24

_HAS_DISPLAY = None
EPOCH = datetime.datetime.utcfromtimestamp(0)

class ReasonCode:
    '''A class containing constants for predefined order reason codes. Prefer these predefined reason codes if they suit
    the reason you are creating your order.  Otherwise, use your own string.
    '''
    ENTER_LONG = 'enter long'
    ENTER_SHORT = 'enter short'
    EXIT_LONG = 'exit long'
    EXIT_SHORT = 'exit short'
    BACKTEST_END = 'backtest end'
    ROLL_FUTURE = 'roll future'
    NONE = 'none'
    
    # Used for plotting trades
    MARKER_PROPERTIES = {
        ENTER_LONG : {'symbol' : 'P', 'color' : 'blue', 'size' : 50},
        ENTER_SHORT : {'symbol' : 'P', 'color' : 'red', 'size' : 50},
        EXIT_LONG : {'symbol' : 'X', 'color' : 'blue', 'size' : 50},
        EXIT_SHORT : {'symbol' : 'X', 'color' : 'red', 'size' : 50},
        ROLL_FUTURE : {'symbol' : '>', 'color' : 'green', 'size' : 50},
        BACKTEST_END : {'symbol' : '*', 'color' : 'green', 'size' : 50},
        NONE : {'symbol' : 'o', 'color' : 'green', 'size' : 50}
    }
 
def has_display():
    '''
    If we are running in unit test mode or on a server, then don't try to draw graphs, etc.
    '''
    global _HAS_DISPLAY
    if _HAS_DISPLAY is not None: return _HAS_DISPLAY
    
    _HAS_DISPLAY = True
    try:
        plt.figure()
    except:
        _HAS_DISPLAY = False
    return _HAS_DISPLAY


def shift_np(array, n, fill_value = None):
    '''
    Similar to pandas.Series.shift but works on numpy arrays.
    
    Args:
        array: The numpy array to shift
        n: Number of places to shift, can be positive or negative
        fill_value: After shifting, there will be empty slots left in the array.  If set, fill these with fill_value.
          If fill_value is set to None (default), we will fill these with False for boolean arrays, np.nan for floats
    '''
    if array is None: return None
    if len(array) == 0: return array
    
    if fill_value is None:
        fill_value = False if array.dtype == np.dtype(bool) else np.nan

    e = np.empty_like(array)
    if n >= 0:
        e[:n] = fill_value
        e[n:] = array[:-n]
    else:
        e[n:] = fill_value
        e[:n] = array[-n:]
    return e

def set_defaults(df_float_sf = 4, df_display_max_rows = 200, df_display_max_columns = 99, np_seterr = 'raise', plot_style = 'ggplot', mpl_figsize = (8, 6)):
    '''
    Set some display defaults to make it easier to view dataframes and graphs.
    
    Args:
        df_float_sf: Number of significant figures to show in dataframes (default 4). Set to None to use pandas defaults
        df_display_max_rows: Number of rows to display for pandas dataframes when you print them (default 200).  Set to None to use pandas defaults
        df_display_max_columns: Number of columns to display for pandas dataframes when you print them (default 99).  Set to None to use pandas defaults
        np_seterr: Error mode for numpy warnings.  See numpy seterr function for details.  Set to None to use numpy defaults
        plot_style: Style for matplotlib plots.  Set to None to use default plot style.
        mpl_figsize: Default figure size to use when displaying matplotlib plots (default 8,6).  Set to None to use defaults
    '''
    if df_float_sf is not None: pd.options.display.float_format = ('{:.' + str(df_float_sf) + 'g}').format
    if df_display_max_rows is not None: pd.options.display.max_rows = df_display_max_rows
    if df_display_max_columns is not None: pd.options.display.max_columns = df_display_max_columns
    if plot_style is not None: plt.style.use(plot_style)
    if mpl_figsize is not None: mpl.rcParams['figure.figsize'] = mpl_figsize
    if np_seterr is not None: np.seterr(np_seterr)
    pd.options.mode.chained_assignment = None # Turn off bogus 'view' warnings from pandas when modifying dataframes
    try: # This will run if we are in Jupyter
        get_ipython().run_line_magic('matplotlib', 'inline')
    except:
        pass
    plt.rcParams.update({'figure.max_open_warning': 100}) # For unit tests, avoid warning when opening more than 20 figures
    
def str2date(s):
    '''Converts a string like "2008-01-15 15:00:00" to a numpy datetime64.  If s is not a string, return s back'''
    if isinstance(s, str): return np.datetime64(s)
    return s

def strtup2date(tup):
    '''Converts a string tuple like ("2008-01-15", "2009-01-16") to a numpy datetime64 tuple.  
      If the tuple does not contain strings, return it back unchanged'''
    if tup and type(tup) is tuple and isinstance(tup[0], str): return (str2date(tup[0]), str2date(tup[1]))
    return tup

def np_get_index(array, value):
    '''Get index of a value in a numpy array.  Returns -1 if the value does not exist.'''
    x = np.where(array == value)
    if len(x[0]): return x[0][0]
    return -1

def date_2_num(d):
    '''
    Adopted from matplotlib.mdates.date2num so we don't have to add a dependency on matplotlib here
    '''
    extra = d - d.astype('datetime64[s]').astype(d.dtype)
    extra = extra.astype('timedelta64[ns]')
    t0 = np.datetime64('0001-01-01T00:00:00').astype('datetime64[s]')
    dt = (d.astype('datetime64[s]') - t0).astype(np.float64)
    dt += extra.astype(np.float64) / 1.0e9
    dt = dt / SEC_PER_DAY + 1.0

    NaT_int = np.datetime64('NaT').astype(np.int64)
    d_int = d.astype(np.int64)
    try:
        dt[d_int == NaT_int] = np.nan
    except TypeError:
        if d_int == NaT_int:
            dt = np.nan
    return dt

def resample_ohlc(dates, o, h, l, c, v, sampling_frequency):
    '''Downsample OHLCV data using sampling frequency
    
    Args:
        o: open price, downsampling uses the first point in the bin
        h: high price, downsampling uses the max
        l: low price, downsampling uses the min
        c: close price, downsampling uses the last point
        v: volume, downsampling uses the sum
        sampling_frequency: See pandas frequency strings
        
    Returns:
        A tuple of arrays, corresponding to each array passed in that was not None.  
          For example, if l and v were passed in as None, the tuple will not contain these.
        
    >>> dates = np.array(['2018-01-08 15:00:00', '2018-01-09 15:00:00', '2018-01-09 15:00:00', '2018-01-11 15:00:00'], dtype = 'M8[ns]')
    >>> o = np.array([8.9, 9.1, 9.3, 8.6])
    >>> h = np.array([9.0, 9.3, 9.4, 8.7])
    >>> l = np.array([8.8, 9.0, 9.2, 8.4])
    >>> c = np.array([8.95, 9.2, 9.35, 8.5])
    >>> v = np.array([200, 100, 150, 300])
    >>> resample_ohlc(dates, o, h, l, c, None, sampling_frequency = 'D')
    (array(['2018-01-08T00:00:00.000000000', '2018-01-09T00:00:00.000000000',
            '2018-01-10T00:00:00.000000000', '2018-01-11T00:00:00.000000000'], dtype='datetime64[ns]'), 
            array([8.9, 9.1, nan, 8.6]), array([9. , 9.4, nan, 8.7]), array([8.8, 9. , nan, 8.4]), array([8.95, 9.35,  nan, 8.5 ]), None)
    '''
    if sampling_frequency is None: return dates, o, h, l, c, v
    df = pd.DataFrame({'date' : dates, 'o' : o, 'h' : h, 'l' : l, 'c' : c, 'v' : v}).set_index('date')
    df = df.resample(sampling_frequency).agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v' : 'sum'}).dropna(how = 'all')
    
    col_list = [df.index.values]
    for col in [(o, 'o'), (h, 'h'), (l, 'l'), (c, 'c'), (v, 'v')]:
        col_list.append(None if col[0] is None else df[col[1]].values)
    return tuple(col_list)

def resample_ts(dates, values, sampling_frequency):
    '''Downsample a pair of dates and values using sampling frequency, using the last value if it does not exist at bin edge.  See pandas.Series.resample
    
    Args:
        dates: a numpy datetime64 array
        values: a numpy array
        sampling_frequency: See pandas frequency strings
    '''
    if sampling_frequency is None: return dates, values
    s = pd.Series(values, index = dates).resample(sampling_frequency).last()
    return s.index.values, s.values

def zero_to_nan(array):
    '''Converts any zeros in a numpy array to nans'''
    if array is None: return None
    return np.where(array == 0, np.nan, array)

def nan_to_zero(array):
    '''Converts any nans in a numpy float array to 0'''
    if array is None: return None
    return np.where(np.isnan(array), 0, array)

def monotonically_increasing(array):
    '''
    Returns True if the array is monotonically_increasing, False otherwise
    
    >>> monotonically_increasing(np.array(['2018-01-02', '2018-01-03'], dtype = 'M8[D]'))
    True
    >>> monotonically_increasing(np.array(['2018-01-02', '2018-01-02'], dtype = 'M8[D]'))
    False
    '''
    if not len(array): return False
    return np.all(np.diff(array).astype(np.float) > 0)

def infer_frequency(dates):
    '''Returns most common frequency of date differences as a fraction of days
    Args:
        dates: A numpy array of monotonically increasing datetime64
    >>> dates = np.array(['2018-01-01 11:00:00', '2018-01-01 11:15:00', '2018-01-01 11:30:00', '2018-01-01 11:35:00'], dtype = 'M8[ns]')
    >>> infer_frequency(dates)
    0.01041667
    '''
    assert(monotonically_increasing(dates))
    numeric_dates = date_2_num(dates)
    diff_dates = np.round(np.diff(numeric_dates), 8)
    (values,counts) = np.unique(diff_dates, return_counts=True)
    ind = np.argmax(counts)
    return diff_dates[ind]

def series_to_array(series):
    '''Convert a pandas series to a numpy array.  If the object is not a pandas Series return it back unchanged'''
    if type(series) == pd.Series: return series.values
    return series

def to_csv(df, file_name, index = False, compress = False, *args, **kwargs):
    """
    Creates a temporary file then renames to the permanent file so we don't have half written files.
    Also optionally compresses using the xz algorithm
    """
    compression = None
    suffix = ''
    if compress:
        compression = 'xz'
        suffix = '.xz'
    df.to_csv(file_name + '.tmp', index = index, compression = compression, *args, **kwargs)
    os.rename(file_name + '.tmp', file_name + suffix)
    
def millis_since_epoch(dt):
    """
    Given a python datetime, return number of milliseconds between the unix epoch and the datetime.
    Returns a float since it can contain fractions of milliseconds as well
    >>> millis_since_epoch(datetime.datetime(2018, 1, 1))
    1514764800000.0
    """
    return (dt - EPOCH).total_seconds() * 1000.0

def infer_compression(input_filename):
    """
    Infers compression for a file from its suffix.  For example, given "/tmp/hello.gz", this will return "gzip"
    >>> infer_compression("/tmp/hello.gz")
    'gzip'
    >>> infer_compression("/tmp/abc.txt") is None
    True
    """
    parts = input_filename.split('.')
    if len(parts) <= 1: return None
    suffix = parts[-1]
    if suffix == 'gz': return 'gzip'
    if suffix == 'bz2': return 'bz2'
    if suffix =='zip': return 'zip'
    if suffix == 'xz': return 'xz'
    return None


def touch(fname, mode=0o666, dir_fd=None, **kwargs):
    '''replicate unix touch command, i.e create file if it doesn't exist, otherwise update timestamp'''
    flags = os.O_CREAT | os.O_APPEND
    with os.fdopen(os.open(fname, flags=flags, mode=mode, dir_fd=dir_fd)) as f:
        os.utime(f.fileno() if os.utime in os.supports_fd else fname,
            dir_fd=None if os.supports_fd else dir_fd, **kwargs)
        
def is_newer(filename, ref_filename):
    '''whether filename ctime (modfication time) is newer than ref_filename or either file does not exist
    >>> import time
    >>> import tempfile
    >>> temp_dir = tempfile.gettempdir()
    >>> touch(f'{temp_dir}/x.txt')
    >>> time.sleep(0.1)
    >>> touch(f'{temp_dir}/y.txt')
    >>> is_newer(f'{temp_dir}/y.txt', f'{temp_dir}/x.txt')
    True
    >>> touch(f'{temp_dir}/y.txt')
    >>> time.sleep(0.1)
    >>> touch(f'{temp_dir}/x.txt')
    >>> is_newer(f'{temp_dir}/y.txt', f'{temp_dir}/x.txt')
    False
    ''' 
    if not os.path.isfile(filename) or not os.path.isfile(ref_filename): return True
    return os.path.getmtime(filename) > os.path.getmtime(ref_filename)

