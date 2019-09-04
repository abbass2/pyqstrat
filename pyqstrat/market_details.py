#cell 0
import numpy as np
import datetime
import calendar as cal
from pyqstrat.holiday_calendars import Calendar
from pyqstrat.pq_utils import decode_future_code

def third_friday_of_month(calendar, month, year, roll = 'backward'):
    '''
    >>> calendar = Calendar.get_calendar(Calendar.NYSE)
    >>> third_friday_of_month(calendar, 3, 2017)
    numpy.datetime64('2017-03-17')
    '''
    # From https://stackoverflow.com/questions/18424467/python-third-friday-of-a-month
    FRIDAY = 4
    first_day_of_month = datetime.datetime(year, month, 1)
    first_friday = first_day_of_month + datetime.timedelta(days=((FRIDAY - cal.monthrange(year, month)[0]) +7) %7 )
    # 4 is friday of week
    third_friday = first_friday + datetime.timedelta(days=14)
    third_friday = third_friday.date()
    third_friday = calendar.add_trading_days(third_friday, 0, roll)
    return third_friday


FUTURE_CODES_INT = {'F' : 1, 'G' : 2, 'H' : 3, 'J' : 4, 'K' : 5, 'M' : 6, 'N' : 7, 'Q' : 8, 'U' : 9, 'V' : 10, 'X' : 11, 'Z' : 12}
FUTURES_CODES_INVERTED = dict([[v,k] for k,v in FUTURE_CODES_INT.items()])

FUTURE_CODES_STR = {'F' : 'jan', 'G' : 'feb', 'H' : 'mar', 'J' : 'apr', 'K' : 'may', 'M' : 'jun', 'N' : 'jul', 'Q' : 'aug', 'U' : 'sep', 'V' : 'oct', 'X' : 'nov', 'Z' : 'dec'}

def decode_future_code(future_code, as_str = True):
    '''
    Given a future code such as "X", return either the month number (from 1 - 12) or the month abbreviation, such as "nov"
    
    Args:
        future_code (str): the one letter future code
        as_str (bool, optional): If set, we return the abbreviation, if not, we return the month number
        
    >>> decode_future_code('X', as_str = False)
    11
    >>> decode_future_code('X')
    'nov'
    '''
    
    if len(future_code) != 1: raise Exception("Future code must be a single character")
    if as_str:
        if future_code not in FUTURE_CODES_STR: raise Exception(f'unknown future code: {future_code}')
        return FUTURE_CODES_STR[future_code]
    
    if future_code not in FUTURE_CODES_INT: raise Exception(f'unknown future code: {future_code}')
    return FUTURE_CODES_INT[future_code]

def get_fut_code(month):
    '''
    Given a month number such as 3 for March, return the future code for it, e.g. H
    >>> get_fut_code(3)
    'H'
    '''
    return FUTURES_CODES_INVERTED[month]

def get_fut_expiry(calendar, fut_symbol):
    '''
    >>> calendar = Calendar.get_calendar(Calendar.NYSE)
    >>> get_fut_expiry(calendar, 'ESH8')
    numpy.datetime64('2018-03-16T08:30')
    '''
    assert fut_symbol.startswith('ES'), f'unknown future type: {fut_symbol}'
    month_str = fut_symbol[-2:-1]
    year_str = fut_symbol[-1:]
    month = decode_future_code(month_str, as_str = False)
    year = 2010 + int(year_str)
    expiry_date = third_friday_of_month(calendar, month, year).astype(datetime.date)
    return np.datetime64(expiry_date) + np.timedelta64(8 * 60 + 30, 'm')

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags = doctest.NORMALIZE_WHITESPACE)

#cell 1


