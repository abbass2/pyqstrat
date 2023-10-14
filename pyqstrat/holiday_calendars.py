# $$_ Lines starting with # $$_* autogenerated by jup_mini. Do not modify these
# $$_code
# $$_ %%checkall
from __future__ import annotations
import numpy as np
import pandas as pd
from collections.abc import Iterable
import datetime
# import os
# import inspect
import calendar as cal
import dateutil.relativedelta as rd
import pandas_market_calendars as mcal
# from types import FrameType
from typing import Union
from pyqstrat.pq_utils import assert_

DateTimeType = Union[pd.Timestamp, str, np.datetime64, datetime.datetime, datetime.date]


def _as_np_date(val: DateTimeType) -> np.datetime64 | np.ndarray | None:
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
        np_date = np.datetime64(val).astype('M8[D]')  # type: ignore
        if isinstance(np_date.astype(datetime.datetime), int):  # User can pass in a string like 20180101 which gets parsed as a year
            raise Exception(f'invalid date: {val}')
        return np_date
    if isinstance(val, pd.Timestamp): 
        return val.to_datetime64().astype('M8[D]')
    if isinstance(val, pd.Series) or isinstance(val, pd.DatetimeIndex):
        return val.values.astype('M8[D]')
    if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.datetime64): 
        return val.astype('M8[D]')
    return None


def _normalize_datetime(val: DateTimeType) -> tuple[np.datetime64, np.timedelta64]:
    '''
    Break up a datetime into numpy date and numpy timedelta.  
    
    Args:
        val: The datetime to normalize.  Can be an array or a single datetime as a string, pandas timestamp, numpy datetime 
            or python date or datetime
    
    >>> date, td = _normalize_datetime(pd.Timestamp('2016-05-01 3:55:00'))
    >>> assert date == np.datetime64('2016-05-01') and td == np.timedelta64(14100000000000,'ns')
    >>> date, td = _normalize_datetime('2016-05-01')
    >>> assert date == np.datetime64('2016-05-01') and td == np.timedelta64(0, 'D')
    >>> x = pd.DataFrame({'x' : [np.datetime64('2015-01-01 05:00:00'), np.datetime64('2015-02-01 06:00:00')]})
    >>> dates, tds = _normalize_datetime(x.x)
    >>> assert (all(dates == np.array(['2015-01-01', '2015-02-01'], dtype='datetime64[D]'))
    ...    and all(tds == np.array([18000000000000, 21600000000000], dtype='timedelta64[ns]')))
    >>> x = pd.DataFrame({'x' : [1, 2]}, index = [np.datetime64('2015-01-01 05:00:00'), np.datetime64('2015-02-01 06:00:00')])
    >>> dates, tds = _normalize_datetime(x.index)
    >>> assert (all(dates == np.array(['2015-01-01', '2015-02-01'], dtype='datetime64[D]'))
    ...    and all(tds == np.array([18000000000000, 21600000000000], dtype='timedelta64[ns]')))
    '''
    if isinstance(val, pd.Timestamp): 
        dtime = val.to_datetime64()
    elif isinstance(val, pd.Series) or isinstance(val, pd.DatetimeIndex):
        dtime = val.values
    elif isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.datetime64): 
        dtime = val
    else:
        dtime = np.datetime64(val)  # type: ignore
        
    date = dtime.astype('M8[D]')
    time_delta = dtime - date
    return date, time_delta


def _normalize(start: DateTimeType,
               end: DateTimeType,
               include_first: bool,
               include_last: bool) -> tuple[np.datetime64, np.datetime64] | tuple[np.ndarray, np.ndarray]:
    '''
    Given a start and end date, return a new start and end date, taking into account include_first and include_last flags
    
    Args:
        start: start date
        end: end_date
        include_first: whether to increment start date by 1
        include_last: whether to increment end date by 1

    Return:
        new start and end dates
        
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
    
    assert_(s is not None and e is not None)
     
    if not include_first and s is not None:
        s += np.timedelta64(1, 'D')

    if include_last and e is not None:
        e += np.timedelta64(1, 'D')

    return s, e  # type: ignore


class Calendar:
    
    _bus_day_calendars: dict[str, np.busdaycalendar] = {}
    
    def __init__(self, calendar_name: str) -> None:
        '''
        Create a calendar object
        Args:
            calendar_name (str): name of calendar as defined in the pandas_market_calendars package
        '''
        if calendar_name not in Calendar._bus_day_calendars:
            cal = mcal.get_calendar(calendar_name)
            holidays = cal.holidays()
            _holidays = np.array([hol for hol in holidays.holidays])
            Calendar._bus_day_calendars[calendar_name] = np.busdaycalendar(holidays=_holidays)
        self.bus_day_cal = Calendar._bus_day_calendars[calendar_name]
        
    def is_trading_day(self, dates: DateTimeType) -> bool | np.ndarray:
        '''
        Returns whether the date is not a holiday or a weekend
        
        Args:
            dates: date times to check
        Return:
            Whether this date is a trading day
        
        >>> import datetime
        >>> eurex = Calendar('EUREX')
        >>> eurex.is_trading_day('2016-12-25')
        False
        >>> eurex.is_trading_day(datetime.date(2016, 12, 22))
        True
        >>> nyse = Calendar('NYSE')
        >>> nyse.is_trading_day('2017-04-01') # Weekend
        False
        >>> nyse.is_trading_day(np.arange('2017-04-01', '2017-04-09', dtype = np.datetime64)) # doctest:+ELLIPSIS
        array([False, False,  True,  True,  True,  True,  True, False]...)
        '''
        if isinstance(dates, str) or isinstance(dates, datetime.date): 
            dates = np.datetime64(dates, 'D')  # type: ignore
            if isinstance(dates.astype(datetime.datetime), int):  # user can pass in a string like 20180101 which gets parsed as a date
                raise Exception(f'invalid date: {dates}')
        if isinstance(dates, pd.Series): dates = dates.values
        return np.is_busday(dates.astype('M8[D]'), busdaycal=self.bus_day_cal)
    
    def num_trading_days(self, 
                         start: DateTimeType,
                         end: DateTimeType,
                         include_first: bool = False,
                         include_last: bool = True) -> float | np.ndarray:
        '''
        Count the number of trading days between two date series including those two dates
        
        >>> eurex = Calendar('EUREX')
        >>> eurex.num_trading_days('2009-01-01', '2011-12-31')
        766.0
        >>> dates = np.arange(np.datetime64('2013-01-01'),np.datetime64('2013-01-09'), np.timedelta64(1, 'D'))
        >>> increments = np.array([5, 0, 3, 9, 4, 10, 15, 29])
        >>> import warnings
        >>> import pandas as pd
        >>> warnings.filterwarnings(action = 'ignore', category = pd.errors.PerformanceWarning)
        >>> dates2 = dates + increments
        >>> dates[4] = np.datetime64('NaT')
        >>> dates2[6] = np.datetime64('NaT')
        >>> df = pd.DataFrame({'x': dates, 'y' : dates2})
        >>> nyse = Calendar('NYSE')
        >>> np.set_printoptions(formatter = {'float' : lambda x : f'{x:.1f}'})  # After numpy 1.13 positive floats don't have a leading space for sign
        >>> print(nyse.num_trading_days(df.x, df.y))
        [3.0 0.0 1.0 5.0 nan 8.0 nan 20.0]
        '''
        iterable = isinstance(start, Iterable) and not isinstance(start, str)
        s_tmp, e_tmp = _normalize(start, end, include_first, include_last)
        # np.busday_count does not like nat dates
        if iterable:
            assert_(isinstance(s_tmp, Iterable))
            # ret = np.full(len(s_tmp), np.nan)  # type: ignore
            # mask = ~(np.isnat(s_tmp) | np.isnat(e_tmp))
            mask = (np.isnat(s_tmp) | np.isnat(e_tmp))
            dummy_date = np.datetime64('1900-01-01')
            s_tmp[mask] = dummy_date  # type: ignore
            e_tmp[mask] = dummy_date  # type: ignore
            count = np.busday_count(s_tmp, e_tmp, busdaycal=self.bus_day_cal).astype(float)  # type: ignore
            count[mask] = np.nan
            return count
        else:
            if np.isnat(s_tmp) or np.isnat(s_tmp): return np.nan
            count = np.busday_count(s_tmp, e_tmp, busdaycal=self.bus_day_cal)  # type: ignore
            return count.astype(float)
        
    def get_trading_days(self,
                         start: DateTimeType,
                         end: DateTimeType,
                         include_first: bool = False,
                         include_last: bool = True) -> int | np.ndarray:
        '''
        Get back a list of numpy dates that are trading days between the start and end
        
        >>> nyse = Calendar('NYSE')
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
    
    def third_friday_of_month(self, month: int, year: int, roll: str = 'backward') -> np.datetime64:
        '''
        >>> nyse = Calendar('NYSE')
        >>> nyse.third_friday_of_month(3, 2017)
        numpy.datetime64('2017-03-17')
        '''
        # From https://stackoverflow.com/questions/18424467/python-third-friday-of-a-month
        FRIDAY = 4
        first_day_of_month = datetime.datetime(year, month, 1)
        first_friday = first_day_of_month + datetime.timedelta(days=((FRIDAY - cal.monthrange(year, month)[0]) + 7) % 7)
        # 4 is friday of week
        third_friday_date = first_friday + datetime.timedelta(days=14)
        third_friday_dt = third_friday_date.date()
        third_friday = self.add_trading_days(third_friday_dt, 0, roll)
        return third_friday  # type: ignore
    
    def add_trading_days(self,
                         start: DateTimeType,
                         num_days: int | np.ndarray, 
                         roll: str = 'raise') -> np.datetime64 | np.ndarray:
        '''
        Adds trading days to a start date
        
        Args:
            start: start datetimes(s)
            num_days: number of trading days to add
            roll: one of 'raise', 'nat', 'forward', 'following', 'backward', 'preceding', 'modifiedfollowing', 'modifiedpreceding' or 'allow'}
                'allow' is a special case in which case, adding 1 day to a holiday will act as if it was not a holiday, and give you the next business day'
                The rest of the values are the same as in the numpy busday_offset function
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
            The datetime num_days trading days after start
            
        >>> calendar = Calendar('NYSE')
        >>> calendar.add_trading_days(datetime.date(2015, 12, 24), 1)
        numpy.datetime64('2015-12-28')
        >>> calendar.add_trading_days(np.datetime64('2017-04-15'), 0, roll = 'preceding') # 4/14/2017 is a Friday and a holiday
        numpy.datetime64('2017-04-13')
        >>> calendar.add_trading_days(np.datetime64('2017-04-08'), 0, roll = 'preceding') # 4/7/2017 is a Friday and not a holiday
        numpy.datetime64('2017-04-07')
        >>> calendar.add_trading_days(np.datetime64('2019-02-17 15:25'), 1, roll = 'allow')
        numpy.datetime64('2019-02-19T15:25')
        >>> calendar.add_trading_days(np.datetime64('2019-02-17 15:25'), -1, roll = 'allow')
        numpy.datetime64('2019-02-15T15:25')
        '''
        start_date, time_delta = _normalize_datetime(start)
        if roll == 'allow':
            # If today is a holiday, roll forward but subtract 1 day so
            num_days = np.where(self.is_trading_day(start) | (num_days < 1), num_days, num_days - 1)  # type: ignore
            roll = 'forward'
        out = np.busday_offset(start_date, num_days, roll=roll, busdaycal=self.bus_day_cal)  # type: ignore
        out = out + time_delta  # for some reason += does not work correctly here.
        return out
        

def get_date_from_weekday(weekday: int, year: int, month: int, week: int) -> np.datetime64:
    '''
    Return the date that falls on a given weekday (Monday = 0) on a week, year and month
    >>> get_date_from_weekday(1, 2019, 10, 4)
    numpy.datetime64('2019-10-22')
    '''
    if week == -1:  # Last day of month
        _, last_day = cal.monthrange(year, month)
        return np.datetime64(datetime.datetime(year, month, last_day)).astype('M8[D]')
    first_day_of_month = datetime.datetime(year, month, 1)
    date = first_day_of_month + rd.relativedelta(weeks=week - 1, weekday=weekday)
    return np.datetime64(date).astype('M8[D]')
    

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
# $$_end_code
