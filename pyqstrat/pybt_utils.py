#cell 0
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

SEC_PER_DAY = 3600 * 24

def set_defaults():
    import numpy as np
    pd.options.display.float_format = '{:.4g}'.format
    pd.options.display.max_rows = 200
    pd.options.display.max_columns = 99
    plt.style.use('ggplot')
    mpl.rcParams['figure.figsize'] = 8, 6
    np.seterr('raise')
    pd.options.mode.chained_assignment = None # Turn off bogus 'view' warnings from pandas when modifying dataframes
    try: # This will run if we are in Jupyter
        get_ipython().run_line_magic('matplotlib', 'inline')
    except:
        pass
    
def str2date(s):
    if isinstance(s, str): return np.datetime64(s)
    return s

def strtup2date(tup):
    if tup and type(tup) is tuple and isinstance(tup[0], str): return (str2date(tup[0]), str2date(tup[1]))
    return tup

def np_get_index(array, value):
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
    '''
    >>> dates = np.array(['2018-01-08 15:00:00', '2018-01-09 15:00:00', '2018-01-09 15:00:00', '2018-01-11 15:00:00'], dtype = 'M8[ns]')
    >>> o = np.array([8.9, 9.1, 9.3, 8.6])
    >>> h = np.array([9.0, 9.3, 9.4, 8.7])
    >>> l = np.array([8.8, 9.0, 9.2, 8.4])
    >>> c = np.array([8.95, 9.2, 9.35, 8.5])
    >>> v = np.array([200, 100, 150, 300])
    >>> resample_ohlc(dates, o, h, l, None, None, sampling_frequency = 'D')
    (array(['2018-01-08T00:00:00.000000000', '2018-01-09T00:00:00.000000000',
            '2018-01-10T00:00:00.000000000', '2018-01-11T00:00:00.000000000'], dtype='datetime64[ns]'),
     array([ 8.9,  9.1,  nan,  8.6]),
     array([ 9. ,  9.4,  nan,  8.7]),
     array([ 8.8,  9. ,  nan,  8.4]),
     array([ nan,  nan,  nan,  nan]),
     array([0, 0, False, 0], dtype=object))
     '''
    if sampling_frequency is None: return dates, o, h, l, c, v
    df = pd.DataFrame({'date' : dates, 'o' : o, 'h' : h, 'l' : l, 'c' : c, 'v' : v}).set_index('date')
    df = df.resample(sampling_frequency).agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v' : 'sum'}).dropna(how = 'all')
    return df.index.values, df.o.values, df.h.values, df.l.values, df.c.values, df.v.values

def resample_ts(dates, values, sampling_frequency):
    if sampling_frequency is None: return dates, values
    s = pd.Series(values, index = dates).resample(sampling_frequency).last()
    return s.index.values, s.values

def zero_to_nan(array):
    if array is None: return None
    return np.where(array == 0, np.nan, array)

def nan_to_zero(array):
    if array is None: return None
    return np.where(np.isnan(array), 0, array)

def monotonically_increasing(array):
    '''
    >>> monotonically_increasing(np.array(['2018-01-02', '2018-01-03'], dtype = 'M8[D]'))
    True
    >>> monotonically_increasing(np.array(['2018-01-02', '2018-01-02'], dtype = 'M8[D]'))
    False
    '''
    if not len(array): return False
    return np.all(np.diff(array).astype(np.float) > 0)

def infer_frequency(dates):
    '''
    Returns frequency of closest points as number of days including fractions of days
    '''
    num_dates = date_2_num(dates)
    freq = np.nanmin(np.diff(num_dates))
    if freq <= 0: raise Exception('could not infer date frequency')
    return freq

