import re
import datetime
import calendar as cal
import dateutil.relativedelta as rd
import numpy as np
from pyqstrat.holiday_calendars import Calendar

_cme_calendar = Calendar.get_calendar(Calendar.NYSE)

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

def get_future_code(month):
    '''
    Given a month number such as 3 for March, return the future code for it, e.g. H
    >>> get_future_code(3)
    'H'
    '''
    return FUTURES_CODES_INVERTED[month]

def get_future_expiry(underlying: str, fut_symbol: str) -> np.datetime64:
    '''
    >>> assert(get_future_expiry('ES', 'ESH8') == np.datetime64('2018-03-16T08:30'))
    '''
    global _cme_calendar
    assert underlying == 'ES', f'underlying: {underlying} not supported'
    month_str = fut_symbol[-2:-1]
    year_str = fut_symbol[-1:]
    month = decode_future_code(month_str, as_str = False)
    year = 2010 + int(year_str)
    expiry_date = third_friday_of_month(_cme_calendar, month, year).astype(datetime.date)
    return np.datetime64(expiry_date) + np.timedelta64(8 * 60 + 30, 'm')

def decode_option_symbol(name):
    '''
    >>> decode_option_symbol('E1AF8')
    (MO, 2018, 1, 1)
    '''
    if re.match('EW[1-4].[0-9]', name): # Friday
        year = int('201' + name[-1:])
        if year in [2010, 2011]: year += 10
        week = int(name[2:3])
        month = decode_future_code(name[3:4], as_str = False)
        return rd.FR, year, month, week
    if re.match('E[1-5]A.[0-9]', name): # Monday
        year = int('201' + name[-1:])
        if year in [2010, 2011]: year += 10
        week = int(name[1:2])
        month = decode_future_code(name[3:4], as_str = False)
        return rd.MO, year, month, week
    if re.match('E[1-5]C.[0-9]', name): # Wednesday
        year = int('201' + name[-1:])
        if year in [2010, 2011]: year += 10
        week = int(name[1:2])
        month = decode_future_code(name[3:4], as_str = False)
        return rd.WE, year, month, week
    if re.match('EW[A-Z][0-9]', name): # End of month
        year = int('201' + name[-1:])
        if year in [2010, 2011]: year += 10
        week = -1
        month = decode_future_code(name[2:3], as_str = False)
        return rd.WE, year, month, week
    else:
        raise Exception(f'could not decode: {name}')
        
def get_date_from_weekday(weekday, year, month, week):
    if week == -1: # Last day of month
        _, last_day = cal.monthrange(year, month)
        return datetime.date(year, month, last_day)
    first_day_of_month = datetime.date(year, month, 1)
    return first_day_of_month + rd.relativedelta(weeks = week - 1, weekday = weekday)

def get_option_expiry(underlying, symbol):
    '''
    >>> get_option_expiry('ES', 'EW2Z5')
    numpy.datetime64('2015-12-11T15:00')
    >>> get_option_expiry('ES', 'E3AF7')
    numpy.datetime64('2017-01-17T15:00')
    >>> get_option_expiry('ES', 'EWF0')
    numpy.datetime64('2020-01-31T15:00')
    '''
    global _cme_calendar
    assert underlying == 'ES', 'unknown underlying: {underlying}'
    assert ':' not in symbol, f'{symbol} contains : pass in option root instead'
    weekday, year, month, week = decode_option_symbol(symbol)
    expiry = get_date_from_weekday(weekday, year, month, week)
    if weekday in [rd.WE, rd.FR]:
        expiry = _cme_calendar.add_trading_days(expiry, num_days = 0, roll = 'backward')
    else:
        expiry = _cme_calendar.add_trading_days(expiry, num_days = 0, roll = 'forward')
    # Option expirations changed on 9/20/2015 from 3:15 to 3 pm - 
    # See https://www.cmegroup.com/market-regulation/files/15-384.pdf
    expiry += np.where(expiry < np.datetime64('2015-09-20'), np.timedelta64(15 * 60 + 15, 'm'), np.timedelta64(15, 'h')) 
    return expiry

def get_current_future_symbol(underlying : str, curr_date: datetime.date) -> str:
    '''
    >>> assert(get_current_future_symbol('ES', datetime.date(2019, 3, 14)) == 'ESH9')
    >>> assert(get_current_future_symbol('ES', datetime.date(2019, 3, 15)) == 'ESM9')
    '''
    assert underlying == 'ES', f'underlying: {underlying} not supported'
    calendar = Calendar.get_calendar(Calendar.NYSE)
    year = curr_date.year
    month = curr_date.month
    day = curr_date.day
    third_friday = third_friday_of_month(calendar, month, year).astype(datetime.date)
    if month < 3 or (month == 3 and day < third_friday.day): month_str = 'H'
    elif month < 6 or (month == 6 and day < third_friday.day): month_str = 'M'
    elif month < 9 or (month == 9 and day < third_friday.day): month_str = 'U'
    elif month < 12 or (month == 12 and day < third_friday.day): month_str = 'Z'
    else:
        month_str = 'H'
        year += 1
    fut_symbol = 'ES' + month_str + str(year - 2010)
    return fut_symbol

def get_previous_future_symbol(underlying: str, curr_future_symbol: str) -> str:
    '''
    >>> assert(get_previous_future_symbol('ES', 'ESH9') == 'ESZ8')
    '''
    assert underlying == 'ES', f'underlying: {underlying} not supported'
    month = curr_future_symbol[2]
    year = int(curr_future_symbol[3])
    prev_month = {'H': 'Z', 'M': 'H', 'U': 'M', 'Z': 'U'}[month]
    prev_year = year if prev_month != 'Z' else year - 1
    if prev_year == -1: prev_year == 9
    return f'ES{prev_month}{prev_year}'


def get_next_future_symbol(underlying: str, curr_future_symbol: str) -> str:
    '''
    >>> assert(get_next_future_symbol('ES', 'ESZ8') == 'ESH9')
    '''
    assert underlying == 'ES', f'underlying: {underlying} not supported'
    month = curr_future_symbol[2]
    year = int(curr_future_symbol[3])
    next_month = {'Z': 'H', 'H': 'M', 'M': 'U', 'U': 'Z'}[month]
    next_year = year if next_month != 'H' else year + 1
    if next_year == 10: next_year == 0
    return f'ES{next_month}{next_year}'

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags = doctest.NORMALIZE_WHITESPACE)

