
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import collections
import datetime
import os
import inspect

def _as_np_date(val):
    '''
    Convert a pandas timestamp, string, np.datetime64('M8[ns]'), datetime, date to a numpy datetime64 [D] and remove time info.
    Returns None if the value cannot be converted
    >>> _as_np_date(pd.Timestamp('2016-05-01 3:55:00'))
    numpy.datetime64('2016-05-01')
    >>> _as_np_date('2016-05-01')
    numpy.datetime64('2016-05-01')
    >>> x = pd.DataFrame({'x' : [np.datetime64('2015-01-01 05:00:00'), np.datetime64('2015-02-01 06:00:00')]})
    >>> _as_np_date(x.x)
    array(['2015-01-01', '2015-02-01'], dtype='datetime64[D]')
    >>> _as_np_date(pd.Series([np.datetime64('2015-01-01 05:00:00'), np.datetime64('2015-02-01 06:00:00')]))
    array(['2015-01-01', '2015-02-01'], dtype='datetime64[D]')
    >>> x = pd.DataFrame({'x' : [1, 2]}, index = [np.datetime64('2015-01-01 05:00:00'), np.datetime64('2015-02-01 06:00:00')])
    >>> _as_np_date(x.index)
    array(['2015-01-01', '2015-02-01'], dtype='datetime64[D]')
    '''
    if isinstance(val, np.datetime64): 
        return val.astype('M8[D]')
    if isinstance(val, str) or isinstance(val, datetime.date) or isinstance(val, datetime.datetime):
        np_date = np.datetime64(val).astype('M8[D]')
        if isinstance(np_date.astype(datetime.datetime), int): # User can pass in a string like 20180101 which gets parsed as a year
            raise Exception(f'invalid date: {val}')
        return np_date
    if isinstance(val, pd.Timestamp): 
        return timestamp.to_datetime64().astype('M8[D]')
    if isinstance(val, pd.Series) or isinstance(val, pd.DatetimeIndex):
        return val.values.astype('M8[D]')
    if isinstance(val, np.ndarray) and val.dtype == np.dtype('M8[ns]'): 
        return val.astype('M8[D]')
    return None


def _normalize(start, end, include_first, include_last):
    '''
    Given a start and end date, return a new start and end date, taking into account include_first and include_last flags
    
    Args:
        start: start date, can be string or datetime or np.datetime64
        end: end_date, can be string or datetime or np.datetime64
        include_first (bool): whether to increment start date by 1
        include_last (bool): whether to increment end date by 1

    Return:
        np.datetime64, np.datetime64 : new start and end dates
        
    >>> x = np.arange(np.datetime64('2017-01-01'), np.datetime64('2017-01-10'))
    >>> y = x.astype('M8[ns]')
    >>> xcp = x.copy()
    >>> ycp = y.copy()
    >>> _normalize(x, y, False, False) # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    (array(...
    >>> (x == xcp).all()
    True
    >>> (y == ycp).all()
    True
    '''
    s = _as_np_date(start)
    e = _as_np_date(end)
    
    if s is None: s = start.copy()
    if e is None: e = end.copy()

    if not include_first:
        s += np.timedelta64(1, 'D')

    if include_last:
        e += np.timedelta64(1, 'D')

    return s,e

def read_holidays(calendar_name, dirname = None):
    '''
    Reads a csv with a holidays column containing holidays (not including weekends)
    '''
    if dirname is None: dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    
    if not os.path.isdir(dirname + '/refdata'):
        #if os.path.isdir(dirname + '/pyqstrat/refdata'):
        #    dirname = dirname + '/pyqstrat'
        #else:
        #    raise Exception(f'path {dirname}/refdata and {dirname}/pyqstrat/refdata do not exist')
        raise Exception(f'path {dirname}/refdata does not exist')
    df = pd.read_csv(f'{dirname}/refdata/holiday_calendars/{calendar_name}.csv')
    holidays = pd.to_datetime(df.holidays, format='%Y-%m-%d').values.astype('M8[D]')
    return holidays

class Calendar(object):
    
    NYSE = "nyse"
    EUREX = "eurex"
    _calendars = {}
    
    def __init__(self, holidays):
        '''
        Do not use this function directly.  Use Calendar.get_calendar instead
        Args:
            holidays (np.array of datetime64[D]): holidays for this calendar, excluding weekends
        '''
        self.bus_day_cal = np.busdaycalendar(holidays = holidays)
        
    def is_trading_day(self, dates):
        '''
        Returns whether the date is not a holiday or a weekend
        
        Args:
            dates: str or datetime.datetime or np.datetime64[D] or numpy array of np.datetime64[D]
        Return:
            bool: Whether this date is a trading day
        
        >>> import datetime
        >>> eurex = Calendar.get_calendar(Calendar.EUREX)
        >>> eurex.is_trading_day('2016-12-25')
        False
        >>> eurex.is_trading_day(datetime.date(2016, 12, 22))
        True
        >>> nyse = Calendar.get_calendar(Calendar.NYSE)
        >>> nyse.is_trading_day('2017-04-01') # Weekend
        False
        >>> nyse.is_trading_day(np.arange('2017-04-01', '2017-04-09', dtype = np.datetime64)) # doctest:+ELLIPSIS
        array([False, False,  True,  True,  True,  True,  True, False]...)
        '''
        if isinstance(dates, str) or isinstance(dates, datetime.date): 
            dates = np.datetime64(dates)
            if isinstance(dates.astype(datetime.datetime), int): # User can pass in a string like 20180101 which gets parsed as a year
                raise Exception(f'invalid date: {dates}')
        return np.is_busday(dates, busdaycal = self.bus_day_cal)
    
    def num_trading_days(self, start, end, include_first = False, include_last = True):
        '''
        Count the number of trading days between two date series including those two dates
        You can pass in a string like '2009-01-01' or a python date or a pandas series for 
        start and end
        
        >>> eurex = Calendar.get_calendar(Calendar.EUREX)
        >>> eurex.num_trading_days('2009-01-01', '2011-12-31')
        772.0
        >>> dates = pd.date_range('20130101',periods=8)
        >>> increments = np.array([5, 0, 3, 9, 4, 10, 15, 29])
        >>> dates2 = dates + increments * 1000000000000000
        >>> df = pd.DataFrame({'x': dates, 'y' : dates2})
        >>> df.iloc[4]['x'] = np.nan
        >>> df.iloc[6]['y'] = np.nan
        >>> nyse = Calendar.get_calendar(Calendar.NYSE)
        >>> np.set_printoptions(formatter = {'float' : lambda x : f'{x:.1f}'})  # After numpy 1.13 positive floats don't have a leading space for sign
        >>> print(nyse.num_trading_days(df.x, df.y))
        [39.0 0.0 23.0 71.0 nan 80.0 nan 232.0]
        '''
        s_tmp, e_tmp = _normalize(start, end, include_first, include_last)
        # np.busday_count does not like nat dates so convert them to known numbers and then check for them.
        DUMMY_BEGIN_DATE = np.datetime64('1700-01-01')
        DUMMY_END_DATE = np.datetime64('1829-03-08')
        DUMMY_DIFF = 33701.
        s = np.where((np.isnat(s_tmp) | np.isnat(e_tmp)), DUMMY_BEGIN_DATE, s_tmp)
        e = np.where((np.isnat(s_tmp) | np.isnat(e_tmp)), DUMMY_END_DATE, e_tmp)
        count = np.busday_count(s, e, busdaycal = self.bus_day_cal).astype(np.float)
        count = np.where(count == DUMMY_DIFF, np.nan, count)
        if isinstance(start, str) and isinstance(end, str):
            return np.ndarray.item(count)
        if not isinstance(start, collections.Iterable) and not isinstance(end, collections.Iterable):
            return np.ndarray.item(count)
        return count
        
    def get_trading_days(self, start, end, include_first = False, include_last = True):
        '''
        Get back a list of numpy dates that are trading days between the start and end
        
        >>> nyse = Calendar.get_calendar(Calendar.NYSE)
        >>> nyse.get_trading_days('2005-01-01', '2005-01-08')
        array(['2005-01-03', '2005-01-04', '2005-01-05', '2005-01-06', '2005-01-07'], dtype='datetime64[D]')
        >>> nyse.get_trading_days(datetime.date(2005, 1, 1), datetime.date(2005, 2, 1))
        array(['2005-01-03', '2005-01-04', '2005-01-05', '2005-01-06',
               '2005-01-07', '2005-01-10', '2005-01-11', '2005-01-12',
               '2005-01-13', '2005-01-14', '2005-01-18', '2005-01-19',
               '2005-01-20', '2005-01-21', '2005-01-24', '2005-01-25',
               '2005-01-26', '2005-01-27', '2005-01-28', '2005-01-31', '2005-02-01'], dtype='datetime64[D]')
        >>> nyse.get_trading_days(datetime.date(2016, 1, 5), datetime.date(2016, 1, 29), include_last = False)
        array(['2016-01-06', '2016-01-07', '2016-01-08', '2016-01-11',
               '2016-01-12', '2016-01-13', '2016-01-14', '2016-01-15',
               '2016-01-19', '2016-01-20', '2016-01-21', '2016-01-22',
               '2016-01-25', '2016-01-26', '2016-01-27', '2016-01-28'], dtype='datetime64[D]')
        >>> nyse.get_trading_days('2017-07-04', '2017-07-08', include_first = False)
        array(['2017-07-05', '2017-07-06', '2017-07-07'], dtype='datetime64[D]')
        >>> nyse.get_trading_days(np.datetime64('2017-07-04'), np.datetime64('2017-07-08'), include_first = False)
        array(['2017-07-05', '2017-07-06', '2017-07-07'], dtype='datetime64[D]')
        ''' 
        s, e = _normalize(start, end, include_first, include_last)
        dates = np.arange(s, e, dtype='datetime64[D]')
        dates = dates[np.is_busday(dates, busdaycal=self.bus_day_cal)]
        return dates
    
    def add_trading_days(self, start, num_days, roll = 'raise'):
        '''
        Adds trading days to a start date
        
        Args:
            start: np.datetime64 or str or datetime
            num_days (int): number of trading days to add
            roll (str, optional): one of 'raise', 'nat', 'forward', 'following', 'backward', 'preceding', 'modifiedfollowing', 'modifiedpreceding'}
                From numpy documentation: 
                How to treat dates that do not fall on a valid day. The default is ‘raise’.
                'raise' means to raise an exception for an invalid day.
                'nat' means to return a NaT (not-a-time) for an invalid day.
                'forward' and 'following’ mean to take the first valid day later in time.
                'backward' and 'preceding' mean to take the first valid day earlier in time.
                'modifiedfollowing' means to take the first valid day later in time unless it is across a Month boundary, 
                in which case to take the first valid day earlier in time.
                'modifiedpreceding' means to take the first valid day earlier in time unless it is across a Month boundary, 
                in which case to take the first valid day later in time.
        
        Return:
            np.datetime64[D]: The date num_days trading days after start
            
        >>> calendar = Calendar.get_calendar(Calendar.NYSE)
        >>> calendar.add_trading_days(datetime.date(2015, 12, 24), 1)
        numpy.datetime64('2015-12-28')
        >>> calendar.add_trading_days(np.datetime64('2017-04-15'), 0, roll = 'preceding') # 4/14/2017 is a Friday and a holiday
        numpy.datetime64('2017-04-13')
        >>> calendar.add_trading_days(np.datetime64('2017-04-08'), 0, roll = 'preceding') # 4/7/2017 is a Friday and not a holiday
        numpy.datetime64('2017-04-07')
        '''
        start = _as_np_date(start)
        out = np.busday_offset(start, num_days, roll = roll, busdaycal = self.bus_day_cal)
        return out
        
    def add_calendar(exchange_name, holidays):
        '''
        Add a trading calendar to the class level calendars dict
        
        Args:
            exchange_name (str): Name of the exchange.
            holidays (np.array of datetime64[D]): holidays for this exchange, excluding weekends
        '''
        Calendar._calendars[exchange_name] = Calendar(holidays)
        
    def get_calendar(exchange_name):
        '''
        Get a calendar object for the given exchange:
        
        Args:
            exchange_name (str): The exchange for which you want a calendar.  Calendar.NYSE, Calendar.EUREX are predefined.
            If you want to add a new calendar, use the add_calendar class level function
        
        Return:
            Calendar: The calendar object
        '''
        
        if exchange_name not in Calendar._calendars:
            if exchange_name not in [Calendar.NYSE, Calendar.EUREX]:
                raise Exception(f'calendar not found: {exchange}')
            holidays = read_holidays(exchange_name)
            Calendar.add_calendar(exchange_name, holidays)
        return Calendar._calendars[exchange_name]
    
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()

