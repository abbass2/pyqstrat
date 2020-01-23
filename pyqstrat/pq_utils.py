import matplotlib as mpl
try:
    import tkinter
except (ImportError, ValueError):
    mpl.use('Agg')  # Support running in headless mode
import matplotlib.pyplot as plt
import os
import tempfile
import asyncio
import datetime
import numpy as np
import logging
import pandas as pd
from typing import Any, Sequence, Optional, Tuple, Callable, MutableSequence, MutableSet, Union

SEC_PER_DAY = 3600 * 24
_HAS_DISPLAY = None
EPOCH = datetime.datetime.utcfromtimestamp(0)
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_FORMAT = '[%(asctime)s.%(msecs)03d %(funcName)s] %(message)s'


def has_display() -> bool:
    '''
    If we are running in unit test mode or on a server, then don't try to draw graphs, etc.
    '''
    global _HAS_DISPLAY
    if _HAS_DISPLAY is not None: return _HAS_DISPLAY
    
    _HAS_DISPLAY = True
    try:
        plt.figure()
    except tkinter.TclError:
        _HAS_DISPLAY = False
    return _HAS_DISPLAY


def shift_np(array: np.ndarray, n: int, fill_value: Any = None) -> np.ndarray:
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


def set_defaults(df_float_sf: int = 8, 
                 df_display_max_rows: int = 200, 
                 df_display_max_columns: int = 99,
                 np_seterr: str = 'raise',
                 plot_style: str = 'ggplot',
                 mpl_figsize: Tuple[int, int] = (8, 6)) -> None:
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
    pd.options.mode.chained_assignment = None  # Turn off bogus 'view' warnings from pandas when modifying dataframes
    plt.rcParams.update({'figure.max_open_warning': 100})  # For unit tests, avoid warning when opening more than 20 figures
    

def str2date(s: Optional[Union[np.datetime64, str]]) -> np.datetime64:
    '''Converts a string like "2008-01-15 15:00:00" to a numpy datetime64.  If s is not a string, return s back'''
    if isinstance(s, str): return np.datetime64(s)
    return s


def strtup2date(tup: Any) -> Tuple[np.datetime64, np.datetime64]:
    '''Converts a string tuple like ("2008-01-15", "2009-01-16") to a numpy datetime64 tuple.  
      If the tuple does not contain strings, return it back unchanged'''
    if tup and type(tup) is tuple and isinstance(tup[0], str): return (str2date(tup[0]), str2date(tup[1]))
    return tup


def remove_dups(l: Sequence[Any], key_func: Callable[[Any], Any] = None) -> MutableSequence[Any]:
    '''
    Remove duplicates from a list 
    Args:
        l: list to remove duplicates from
        key_func: A function that takes a list element and converts it to a key for detecting dups
        
    Returns (List): A list with duplicates removed.  This is stable in the sense that original list elements will retain their order
    
    >>> print(remove_dups(['a', 'd', 'a', 'c']))
    ['a', 'd', 'c']
    >>> print(remove_dups(['a', 'd', 'A']))
    ['a', 'd', 'A']
    >>> print(remove_dups(['a', 'd', 'A'], key_func = lambda e: e.upper()))
    ['a', 'd']
    '''
    new_list = []
    seen: MutableSet[Any] = set() 
    for element in l:
        if key_func:
            key = key_func(element)
        else:
            key = element
        if key not in seen:
            new_list.append(element)
            seen.add(key)
    return new_list


def np_get_index(array: np.ndarray, value: Any) -> int:
    '''Get index of a value in a numpy array.  Returns -1 if the value does not exist.'''
    x = np.where(array == value)
    if len(x[0]): return x[0][0]
    return -1


def np_find_closest(a: np.ndarray, v: Any) -> int:
    '''
    From https://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
    Find index of closest value to array v in array a.  Returns an array of the same size as v
    a must be sorted
    >>> assert(all(np_find_closest(np.array([3, 4, 6]), np.array([4, 2])) == np.array([1, 0])))
    '''
    idx = a.searchsorted(v)
    idx = np.clip(idx, 1, len(a) - 1)
    left = a[idx - 1]
    right = a[idx]
    idx -= v - left < right - v
    return idx


def np_rolling_window(a: np.ndarray, window: int) -> np.ndarray:
    '''
    For applying rolling window functions to a numpy array
    See: https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy
    >>> print(np.std(np_rolling_window(np.array([1, 2, 3, 4]), 2), 1))
    [0.5 0.5 0.5]
    '''
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def np_round(a: np.ndarray, clip: float):
    '''
    Round all elements in an array to the nearest clip
    
    Args:
        a: array with elements to round
        clip: rounding value
    >>> np_round(15.8, 0.25)
    15.75
    '''
        
    return np.round(np.array(a, dtype=np.float) / clip) * clip


def day_of_week_num(a: Union[np.datetime64, np.ndarray]) -> Union[int, np.ndarray]:
    '''
    From https://stackoverflow.com/questions/52398383/finding-day-of-the-week-for-a-datetime64
    Get day of week for a numpy array of datetimes 
    Monday is 0, Sunday is 6
    
    Args:
        a: numpy datetime64 or array of datetime64
        
    Return:
        int or numpy ndarray of int: Monday is 0, Sunday is 6

    >>> day_of_week_num(np.datetime64('2015-01-04'))
    6
    '''
    ret = (a.astype('datetime64[D]').view('int64') - 4) % 7
    if np.isscalar(ret): ret = ret.item()
    return ret


def percentile_of_score(a: np.ndarray) -> np.ndarray:
    '''
    For each element in a, find the percentile of a its in.  From stackoverflow.com/a/29989971/5351549
    Like scipy.stats.percentileofscore but runs in O(n log(n)) time.
    >>> a = np.array([4, 3, 1, 2, 4.1])
    >>> percentiles = percentile_of_score(a)
    >>> assert(all(np.isclose(np.array([ 75.,  50.,   0.,  25., 100.]), percentiles)))
    '''
    assert isinstance(a, np.ndarray), f'expected numpy array, got: {a}'
    if not len(a): return None
    return np.argsort(np.argsort(a)) * 100. / (len(a) - 1)


def date_2_num(d: Union[np.datetime64, np.ndarray]) -> Union[int, np.ndarray]:
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


def resample_vwap(df: pd.DataFrame, sampling_frequency: str) -> np.ndarray:
    '''
    Compute weighted average of vwap given higher frequency vwap and volume
    '''
    if 'v' not in df.columns: return None
    sum_1 = df.vwap * df.v
    sum_2 = sum_1.resample(sampling_frequency).agg(np.sum)
    volume_sum = df.v.resample(sampling_frequency).agg(np.sum)
    vwap = sum_2 / volume_sum
    return vwap


def resample_trade_bars(df, sampling_frequency, resample_funcs=None):
    '''Downsample trade bars using sampling frequency
    
    Args:
        df (pd.DataFrame): Must contain an index of numpy datetime64 type which is monotonically increasing
        sampling_frequency (str): See pandas frequency strings
        resample_funcs (dict of str: int): a dictionary of column name -> resampling function for any columns that are custom defined.  Default None.
            If there is no entry for a custom column, defaults to 'last' for that column
    Returns:
        pd.DataFrame: Resampled dataframe
        
    >>> import math
    >>> df = pd.DataFrame({'date': np.array(['2018-01-08 15:00:00', '2018-01-09 13:30:00', '2018-01-09 15:00:00', '2018-01-11 15:00:00'], dtype = 'M8[ns]'),
    ...          'o': np.array([8.9, 9.1, 9.3, 8.6]), 
    ...          'h': np.array([9.0, 9.3, 9.4, 8.7]), 
    ...          'l': np.array([8.8, 9.0, 9.2, 8.4]), 
    ...          'c': np.array([8.95, 9.2, 9.35, 8.5]),
    ...          'v': np.array([200, 100, 150, 300]),
    ...          'x': np.array([300, 200, 100, 400])
    ...         })
    >>> df['vwap'] =  0.5 * (df.l + df.h)
    >>> df.set_index('date', inplace = True)
    >>> df = resample_trade_bars(df, sampling_frequency = 'D', resample_funcs={'x': lambda df, 
    ...   sampling_frequency: df.x.resample(sampling_frequency).agg(np.mean)})
    >>> assert(len(df) == 4)
    >>> assert(math.isclose(df.vwap.iloc[1], 9.24))
    >>> assert(np.isnan(df.vwap.iloc[2]))
    >>> assert(math.isclose(df.l[3], 8.4))
    '''
    if sampling_frequency is None: return df
    
    if resample_funcs is None: resample_funcs = {}
    if 'vwap' in df.columns: resample_funcs.update({'vwap': resample_vwap})
    
    funcs = {'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v': 'sum'}
    
    agg_dict = {}
    
    for col in df.columns:
        if col in funcs:
            agg_dict[col] = funcs[col]
            continue
        if col not in resample_funcs:
            agg_dict[col] = 'last'
    
    resampled = df.resample(sampling_frequency).agg(agg_dict).dropna(how='all')
    
    for k, v in resample_funcs.items():
        res = v(df, sampling_frequency)
        if res is not None: resampled[k] = res
            
    resampled.reset_index(inplace=True)
    return resampled


def resample_ts(dates: np.ndarray, values: np.ndarray, sampling_frequency: str) -> Tuple[np.ndarray, np.ndarray]:
    '''Downsample a tuple of datetimes and value arrays using sampling frequency, using the last value if it does not exist at the bin edge.
    See pandas.Series.resample
    
    Args:
        dates: a numpy datetime64 array
        values: a numpy array
        sampling_frequency: See pandas frequency strings
        
    Returns:
        Resampled tuple of datetime and value arrays
    '''
    if sampling_frequency is None: return dates, values
    s = pd.Series(values, index=dates).resample(sampling_frequency).last()
    return s.index.values, s.values


def zero_to_nan(array: np.ndarray) -> np.ndarray:
    '''Converts any zeros in a numpy array to nans'''
    if array is None: return None
    return np.where(array == 0, np.nan, array)


def nan_to_zero(array: np.ndarray) -> np.ndarray:
    '''Converts any nans in a numpy float array to 0'''
    if array is None: return None
    return np.where(np.isnan(array), 0, array)


def monotonically_increasing(array: np.ndarray) -> bool:
    '''
    Returns True if the array is monotonically_increasing, False otherwise
    
    >>> monotonically_increasing(np.array(['2018-01-02', '2018-01-03'], dtype = 'M8[D]'))
    True
    >>> monotonically_increasing(np.array(['2018-01-02', '2018-01-02'], dtype = 'M8[D]'))
    False
    '''
    if not len(array): return False
    return np.all(np.diff(array).astype(np.float) > 0)


def infer_frequency(timestamps: np.ndarray) -> float:
    '''Returns most common frequency of date differences as a fraction of days
    Args:
        timestamps: A numpy array of monotonically increasing datetime64
    >>> timestamps = np.array(['2018-01-01 11:00:00', '2018-01-01 11:15:00', '2018-01-01 11:30:00', '2018-01-01 11:35:00'], dtype = 'M8[ns]')
    >>> print(round(infer_frequency(timestamps), 8))
    0.01041667
    '''
    if isinstance(timestamps, pd.Series): timestamps = timestamps.values
    assert(monotonically_increasing(timestamps))
    numeric_dates = date_2_num(timestamps)
    diff_dates = np.round(np.diff(numeric_dates), 8)
    (values, counts) = np.unique(diff_dates, return_counts=True)
    return values[np.argmax(counts)]


def series_to_array(series: pd.Series) -> np.ndarray:
    '''Convert a pandas series to a numpy array.  If the object is not a pandas Series return it back unchanged'''
    if type(series) == pd.Series: return series.values
    return series


def to_csv(df, file_name: str, index: bool = False, compress: bool = False, *args, **kwargs) -> None:
    """
    Creates a temporary file then renames to the permanent file so we don't have half written files.
    Also optionally compresses using the xz algorithm
    """
    compression = None
    suffix = ''
    if compress:
        compression = 'xz'
        suffix = '.xz'
    df.to_csv(file_name + '.tmp', index=index, compression=compression, *args, **kwargs)
    os.rename(file_name + '.tmp', file_name + suffix)
    

def millis_since_epoch(dt: datetime.datetime) -> float:
    """
    Given a python datetime, return number of milliseconds between the unix epoch and the datetime.
    Returns a float since it can contain fractions of milliseconds as well
    >>> millis_since_epoch(datetime.datetime(2018, 1, 1))
    1514764800000.0
    """
    return (dt - EPOCH).total_seconds() * 1000.0


def day_symbol(day_int: Union[int, np.ndarray]) -> Union[str, np.ndarray]:
    day_str = np.select([day_int == 0, day_int == 1, day_int == 2, day_int == 3, day_int == 4, day_int == 5, day_int == 6],
                        ['M', 'Tu', 'W', 'Th', 'F', 'Sa', 'Su'], default='')
    if day_str.shape == (): day_str = np.asscalar(day_str)
    return day_str


def infer_compression(input_filename: str) -> Optional[str]:
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
    if suffix == 'zip': return 'zip'
    if suffix == 'xz': return 'xz'
    return None


def touch(fname: str, mode: int = 0o666, dir_fd: Optional[int] = None, **kwargs) -> None:
    '''replicate unix touch command, i.e create file if it doesn't exist, otherwise update timestamp'''
    flags = os.O_CREAT | os.O_APPEND
    with os.fdopen(os.open(fname, flags=flags, mode=mode, dir_fd=dir_fd)) as f:
        os.utime(f.fileno() if os.utime in os.supports_fd else fname,
                 dir_fd=None if os.supports_fd else dir_fd, **kwargs)
        

def is_newer(filename: str, ref_filename: str) -> bool:
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


def get_empty_np_value(np_dtype: np.dtype) -> Any:
    '''
    Get empty value for a given numpy datatype
    >>> a = np.array(['2018-01-01', '2018-01-03'], dtype = 'M8[D]')
    >>> get_empty_np_value(a.dtype)
    numpy.datetime64('NaT')
    '''
    kind = np_dtype.kind
    if kind == 'f': return np.nan  # float
    if kind == 'b': return False  # bool
    if kind == 'i' or kind == 'u': return 0  # signed or unsigned int
    if kind == 'M': return np.datetime64('NaT')  # datetime
    if kind == 'O' or kind == 'S' or kind == 'U': return ''  # object or string or unicode
    raise Exception(f'unknown dtype: {np_dtype}')
    

def get_temp_dir() -> str:
    if os.access('/tmp', os.W_OK):
        return '/tmp'
    else:
        return tempfile.gettempdir()
    

def linear_interpolate(a1: float, a2: float, x1: float, x2: float, x: float) -> float:
    '''
    >>> print(f'{linear_interpolate(3, 4, 8, 10, 8.9):.3f}')
    3.450
    '''
    return np.where(x2 == x1, np.nan, a1 + (a2 - a1) * (x - x1) / (x2 - x1))


def _add_stream_handler(logger: logging.Logger, log_level: int = logging.INFO, formatter: logging.Formatter = None) -> None:
    if formatter is None: formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)


def get_main_logger() -> logging.Logger:
    main_logger = logging.getLogger('pq')
    if len(main_logger.handlers): return main_logger
    _add_stream_handler(main_logger)
    main_logger.setLevel(logging.INFO)
    main_logger.propagate = False
    return main_logger


def get_child_logger(child_name: str) -> logging.Logger:
    _ = get_main_logger()  # Init handlers if needed
    full_name = 'pq.' + child_name if child_name else 'pq'
    logger = logging.getLogger(full_name)
    return logger


def in_ipython() -> bool:
    '''
    Whether we are running in an ipython (or Jupyter) environment
    '''
    import builtins
    return '__IPYTHON__' in vars(builtins)

    
def async_waitfor(predicate_func: Callable, timeout_secs=5) -> None:
    '''
    Keep yielding until either predicate is true or timeout elapses
    Returns when predicate is True.  If timeout elapses we raise an exception
    '''
    start_time = datetime.datetime.now()
    while True:
        async_yield()
        if predicate_func():
            break
        if (datetime.datetime.now() - start_time).total_seconds() > timeout_secs:
            raise Exception(f'timed out after: {timeout_secs} seconds')
            

def async_sleep(secs: float) -> None:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.sleep(secs))
    

def async_yield() -> None:
    '''
    yield so any other async tasks that are ready can run 
    '''
    async_sleep(0)


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
