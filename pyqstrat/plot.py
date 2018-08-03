
# coding: utf-8

# In[6]:


from functools import reduce
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
import matplotlib.patches as mptch
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import display
from pyqstrat.pq_utils import *


# In[4]:


_VERBOSE = False

class DateFormatter(mtick.Formatter):
    def __init__(self, dates, fmt):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos = 0):
        'Return the label for time x at position pos'
        ind = int(np.round(x))
        if ind >= len(self.dates) or ind < 0: return ''
        return mdates.num2date(self.dates[ind]).strftime(self.fmt)
    
class HorizontalLine:
    def __init__(self, y, name = None, line_type = 'dashed', color = None):
        self.y = y
        self.name = name
        self.line_type = line_type
        self.color = color
    
class DateLine:
    def __init__(self, date, name = None, line_type = 'dashed', color = None):
        self.date = date
        self.name = name
        self.line_type = line_type
        self.color = color
    
class BucketedValues:
    def __init__(self, name, bucket_names, bucket_values, plot_type = 'boxplot', 
                 proportional_widths = True, show_means = True, show_all = True, show_outliers = False, notched = False):
        assert isinstance(bucket_names, list) and isinstance(bucket_values, list) and len(bucket_names) == len(bucket_values)
        if plot_type != 'boxplot':
            raise Exception('only boxplots are currently supported for bucketed values')
        self.name = name
        self.bucket_names = bucket_names
        self.bucket_values = bucket_values
        self.plot_type = plot_type
        self.proportional_widths = proportional_widths
        self.show_means = show_means
        self.show_all = show_all
        self.show_outliers = show_outliers
        self.notched = notched
    
class TimeSeries:
    def __init__(self, name, dates, values, plot_type = 'line', line_type = 'solid', color = None, line_width = None):
        self.name = name
        self.dates = dates
        self.values = values
        self.plot_type = plot_type
        self.line_type = line_type
        self.color = color
        self.line_width = line_width
        
    def reindex(self, all_dates, fill):
        s = pd.Series(self.values, index = self.dates)
        s = s.reindex(all_dates, method = 'ffill' if fill else None)
        self.dates = s.index.values
        self.values = s.values
        
class OHLC:
    def __init__(self, name, dates, o, h, l, c, v = None, plot_type = 'candlestick', colorup='#D5E1DD', colordown='#F2583E'):
        self.name = name
        self.dates = dates
        self.o = o
        self.h = h
        self.l = l
        self.c = c
        self.v = np.empty(len(self.dates), dtype = np.float64) * np.nan if v is None else v
        self.plot_type = plot_type
        self.colorup = colorup
        self.colordown = colordown
        
    def to_df(self):
        return pd.DataFrame({'o' : self.o, 'h' : self.h, 'l' : self.l, 'c' : self.c, 'v' : self.v}, index = self.dates)[['o', 'h', 'l', 'c', 'v']]
        
    def reindex(self, all_dates):
        df = self.to_df()
        df = df.reindex(all_dates)
        self.dates = all_dates
        for col in df.columns:
            setattr(self, col, df[col].values)
                
class TradeSet:
    def __init__(self, name, trades, plot_type = 'scatter', marker = 'P', marker_color = None, marker_size = 50):
        self. name = name
        self.trades = trades
        self.plot_type = plot_type
        self.marker = marker
        self.marker_color = marker_color
        self.marker_size = marker_size
        self.dates = np.array([trade.date for trade in trades], dtype = 'M8[ns]')
        self.values = np.array([trade.price for trade in trades], dtype = np.float)
        
    def reindex(self, all_dates, fill):
        s = pd.Series(self.values, index = self.dates)
        s = s.reindex(all_dates, method = 'ffill' if fill else None)
        self.dates = s.index.values
        self.values = s.values
        
    def __repr__(self):
        s = ''
        for trade in self.trades:
            s += f'{trade.date} {trade.qty} {trade.price}\n'
        return s

def draw_candlestick(ax, index, o, h, l, c, v, colorup='#D5E1DD', colordown='#F2583E'):
                
    width = 0.5
    
    offset = width / 2.0
    lines = []
    patches = []
    
    if not np.isnan(v).all(): # Have to do volume first because of a mpl bug with axes fonts if we use make_axes_locatable after plotting on top axis
        divider = make_axes_locatable(ax)
        vol_ax = divider.append_axes('bottom', size = '25%', sharex = ax)
        _c = np.nan_to_num(c)
        _o = np.nan_to_num(o)
        pos = _c >= _o
        neg = _c < _o
        vol_ax.bar(index[pos], v[pos], color = colorup, width = width)
        vol_ax.bar(index[neg], v[neg], color= colordown, width = width)
    
    for i in index:
        close = c[i]
        open = o[i]
        low = l[i]
        high = h[i]
        
        if close >= open:
            color = colorup
            lower = open
            height = close - open
        else:
            color = colordown
            lower = close
            height = open - close

        vline = mlines.Line2D(
            xdata=(i, i), ydata=(low, high),
            antialiased=True,
            linewidth = 0.75,
            color = 'k'
        )

        rect = mptch.Rectangle(
            xy=(i - offset, lower),
            width=width,
            height=height,
            facecolor=color,
            edgecolor='k', zorder = 10
        )

        lines.append(vline)
        patches.append(rect)
        ax.add_line(vline)
        ax.add_patch(rect)
        
        ax.relim()
        ax.autoscale_view()
        
def draw_boxplot(ax, names, values, proportional_widths = True, notched = False, show_outliers = True, show_means = True, show_all = True):
    outliers = None if show_outliers else ''
    meanpointprops = dict(marker='D')
    assert(isinstance(values, list) and isinstance(names, list) and len(values) == len(names))
    widths = None
    
    if show_all:
        all_values = np.concatenate(values)
        values.append(all_values)
        names.append('all')
    
    if proportional_widths:
        counts = [len(v) for v in values]
        total = float(sum(counts))
        cases = len(counts)
        widths = [c/total for c in counts]  
    
    ax.boxplot(values, notch = notched, sym = outliers, showmeans = show_means, meanprops=meanpointprops, widths = widths) #, widths = proportional_widths);
    ax.set_xticklabels(names);

def _plot_data(ax, data):
    
    if data.plot_type == 'boxplot':
        draw_boxplot(ax, data.bucket_names, data.bucket_values, data.proportional_widths, data.notched, data.show_outliers, data.show_means, data.show_all)
        return None
    
    dates = data.dates
    index = np.arange(len(dates))
    
    line = None
    
    if data.plot_type == 'line':
        line, = ax.plot(index, data.values, linestyle = data.line_type, color = data.color)
    elif data.plot_type == 'scatter':
        line = ax.scatter(index, data.values, marker = data.marker, c = data.marker_color, s = data.marker_size, zorder=100)
    elif data.plot_type == 'bar':
        line = ax.bar(index, data.values, color = data.color)
    elif data.plot_type == 'filled_line':
        values = np.nan_to_num(data.values)
        pos_values = np.where(values > 0, values, 0)
        neg_values = np.where(values < 0, values, 0)
        ax.fill_between(index, pos_values, color='blue', step = 'post', linewidth = 0.0)
        ax.fill_between(index, neg_values, color='red', step = 'post', linewidth = 0.0)
    elif data.plot_type == 'candlestick':
        draw_candlestick(ax, index, data.o, data.h, data.l, data.c, data.v)
    else:
        raise Exception(f'unknown plot type: {data.plot_type}')
    return line

def _draw_date_gap_lines(ax, plot_dates):
    dates = mdates.date2num(plot_dates)
    freq = np.nanmin(np.diff(dates))
    if freq <= 0: raise Exception('could not infer date frequency')
    date_index = np.arange(len(dates))
    date_diff = np.diff(dates)

    for i in date_index:
        if i < len(date_diff) and date_diff[i] > (freq + 0.000000001):
            ax.axvline(x = i + 0.5, linestyle = 'dashed', color = '0.5')
            
def draw_date_line(ax, plot_dates, date, linestyle, color):
    date_index = np.arange(len(plot_dates))
    closest_index = (np.abs(plot_dates - date)).argmin()
    return ax.axvline(x = closest_index, linestyle = linestyle, color = color)

def draw_horizontal_line(ax, y, linestyle, color):
    return ax.axhline(y = y, linestyle = linestyle, color = color)
           
def get_date_formatter(plot_dates, date_format):
    num_dates = mdates.date2num(plot_dates)
    date_range = num_dates[-1] - num_dates[0]
    if date_range > 252:
        date_format = '%d-%b-%Y'
    elif date_range > 7:
        date_format = '%b %d'
    elif date_range > 1:
        date_format = '%d %H:%M'
    else:
        date_format = '%H:%M:%S'
        
    formatter = DateFormatter(num_dates, fmt = date_format)
    return formatter
    
class Subplot:
    def __init__(self, data_list, title = None, date_lines = None, horizontal_lines = None, ylim = None, 
                 height_ratio = None, display_legend = True, legend_loc = 'best', log_y = False, y_tick_format = None):
        if not isinstance(data_list, list): data_list = [data_list]
        self.time_plot = all([not isinstance(data, BucketedValues) for data in data_list])
        if self.time_plot and any([isinstance(data, BucketedValues) for data in data_list]):
            raise Exception('cannot add a non date subplot on a subplot which has time series plots')
        if not self.time_plot and date_lines is not None: 
            raise Exception('date lines can only be specified on a time series subplot')
        self.data_list = data_list
        self.date_lines = [] if date_lines is None else date_lines
        self.horizontal_lines = [] if horizontal_lines is None else horizontal_lines
        self.title = title
        self.ylim = ylim
        self.height_ratio = height_ratio
        self.display_legend = display_legend
        self.legend_loc = legend_loc
        self.log_y = log_y
        self.y_tick_format = y_tick_format
        
    def _resample(self, sampling_frequency):
        dates, values = None, None
        for data in self.data_list:
            values = None
            if isinstance(data, TimeSeries) or isinstance(data, TradeSet):
                data.dates, data.values = resample_ts(data.dates, data.values, sampling_frequency)
            elif isinstance(data, OHLC):
                data.dates, data.o, data.h, data.l, data.c, data.v = resample_ohlc(data.dates, data.o, data.h, data.l, data.c, data.v, sampling_frequency = sampling_frequency)
            else:
                raise Exception(f'unknown type: {data}')
        
    def get_all_dates(self, date_range):
        dates_list = [data.dates for data in self.data_list]
        all_dates = np.array(reduce(np.union1d, dates_list))
        if date_range: all_dates = all_dates[(all_dates >= date_range[0]) & (all_dates <= date_range[1])]
        return all_dates
    
    def _reindex(self, all_dates):
        for data in self.data_list:
            if isinstance(data, OHLC):
                data.reindex(all_dates)
            else:
                fill = not isinstance(data, TradeSet) and not data.plot_type == 'bar'
                data.reindex(all_dates, fill = fill)
            
    def _draw(self, ax, plot_dates, date_formatter):
        
        if self.time_plot:
            self._reindex(plot_dates)
            ax.xaxis.set_major_formatter(date_formatter)
            date_index = np.arange(len(plot_dates))
        
        lines = []
        
        for data in self.data_list:
            if _VERBOSE: print(f'plotting data: {data.name}')
            line  = _plot_data(ax, data)
            lines.append(line)
            
        for date_line in self.date_lines:
            line = draw_date_line(ax, plot_dates, date_line.date, date_line.line_type, date_line.color)
            if date_line.name is not None: lines.append(line)
                
        for horizontal_line in self.horizontal_lines:
            line = draw_horizontal_line(ax, horizontal_line.y, horizontal_line.line_type, horizontal_line.color)
            if horizontal_line.name is not None: lines.append(line)
           
        self.legend_names = [data.name for data in self.data_list]
        self.legend_names += [date_line.name for date_line in self.date_lines if date_line.name is not None]
        self.legend_names += [horizontal_line.name for horizontal_line in self.horizontal_lines if horizontal_line.name is not None]
                
        if self.ylim: ax.set_ylim(self.ylim)
        if (len(self.data_list) > 1 or len(self.date_lines)) and self.display_legend: 
            ax.legend([line for line in lines if line is not None],
                      [self.legend_names[i] for i, line in enumerate(lines) if line is not None], loc = self.legend_loc)
            
        if self.title: ax.set_ylabel(self.title)
 
        if self.log_y: 
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(mtick.AutoLocator())
        if self.y_tick_format:
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter(self.y_tick_format))

class Plot:
    def __init__(self, subplot_list,  title = None, figsize = (20, 15), date_range = None, date_format = None, 
                 sampling_frequency = None, show_grid = True, show_date_gaps = True, hspace = 0.15):
        if isinstance(subplot_list, Subplot): subplot_list = [subplot_list]
        self.subplot_list = subplot_list
        self.title = title
        self.figsize = figsize
        self.date_range = date_range
        self.date_format = date_format
        self.sampling_frequency = sampling_frequency
        self.show_date_gaps = show_date_gaps
        self.show_grid = show_grid
        self.hspace = hspace
        
    def get_plot_dates(self):
        dates_list = []
        for subplot in self.subplot_list:
            if not subplot.time_plot: continue
            subplot._resample(self.sampling_frequency)
            dates_list.append(subplot.get_all_dates(self.date_range))
        if not len(dates_list): return None
        plot_dates = np.array(reduce(np.union1d, dates_list))
        return plot_dates
        
    def draw(self, check_data_size = True):
        
        if not has_display():
            print('no display found, cannot plot')
            return
        
        plot_dates = self.get_plot_dates()
        if check_data_size and len(plot_dates) > 1000:
            raise Exception(f'trying to plot large data set with {len(plot_dates)} points, reduce date range or turn check_data_size flag off')
            
        date_formatter = None
        if plot_dates is not None: 
            date_formatter = get_date_formatter(plot_dates, self.date_format)
        height_ratios = [subplot.height_ratio for subplot in self.subplot_list]
        
        fig = plt.figure(figsize = self.figsize)
        gs = gridspec.GridSpec(len(self.subplot_list), 1, height_ratios= height_ratios, hspace = self.hspace)
        axes = []
        
        for i, subplot in enumerate(self.subplot_list):
            ax = plt.subplot(gs[i])
            axes.append(ax)
            
        time_axes = [axes[i] for i, s in enumerate(self.subplot_list) if s.time_plot]
        if len(time_axes):
            time_axes[0].get_shared_x_axes().join(*time_axes)
            
        for i, subplot in enumerate(self.subplot_list):
            subplot._draw(axes[i], plot_dates, date_formatter)
            
        if self.title: axes[0].set_title(self.title)

        # We may have added new axes in candlestick plot so get list of axes again
        ax_list = fig.axes
        for ax in ax_list:
            if self.show_grid: ax.grid(linestyle='dotted', color = 'grey', which = 'both', alpha = 0.5)
                
        for ax in ax_list:
            if ax not in axes: time_axes.append(ax)
                
        for ax in time_axes:
            if self.show_date_gaps: _draw_date_gap_lines(ax, plot_dates)
            
def select_long_short_trades(trade_pos_list, long_flag, enter_flag):
    trade_list = []
    for v in trade_pos_list:
        trade = v[0]
        pos = v[1]
        prev_pos = pos - trade.qty
        enter_trade = abs(pos) > abs(prev_pos)
        long_trade = trade.qty > 0 if enter_flag else trade.qty < 0
        include = (enter_trade if enter_flag else not enter_trade) and (long_trade if long_flag else not long_trade)
        if include: trade_list.append(trade)
    return trade_list

def get_long_short_trade_sets(trades, positions):
    trade_pos_list = []
    for trade in trades:
        idx = np_get_index(positions[0], trade.date)
        if idx != -1:
            trade_pos_list.append((trade, positions[1][idx]))
 
    return [TradeSet('long_enter', trades = select_long_short_trades(trade_pos_list, True, True), marker = 'P', marker_color = 'b'),
            TradeSet('long_exit', trades = select_long_short_trades(trade_pos_list, True, False), marker = 'X', marker_color = 'b'),
            TradeSet('short_enter', trades = select_long_short_trades(trade_pos_list, False, True), marker = 'P', marker_color = 'r'),
            TradeSet('short_exit', trades = select_long_short_trades(trade_pos_list, False, False), marker = 'X', marker_color = 'r')]

def test_plot():
    
    class MockTrade:
        def __init__(self, date, qty, price):
            self.date = date
            self.qty = qty
            self.price = price
            
        def __repr__(self):
            return f'{self.date} {self.qty} {self.price}'
            
    md_dates = np.array(['2018-01-08 15:00:00', '2018-01-09 15:00:00', '2018-01-10 15:00:00', '2018-01-11 15:00:00'], dtype = 'M8[ns]')
    pnl_dates = np.array(['2018-01-08 15:00:00', '2018-01-09 14:00:00', '2018-01-10 15:00:00', '2018-01-15 15:00:00'], dtype = 'M8[ns]')
    
    positions = (pnl_dates, np.array([0., 5., 0.,-10.]))
    
    trade_dates = np.array(['2018-01-09 14:00:00', '2018-01-10 15:00:00', '2018-01-15 15:00:00'], dtype = 'M8[ns]')
    trade_price = [9., 10., 9.5]
    trade_qty =  [5, -5, -10]
    trades = [MockTrade(trade_dates[i], trade_qty[i], trade_price[i]) for i, d in enumerate(trade_dates)]

    ind_subplot = Subplot([TimeSeries('slow_support', dates = md_dates, values = np.array([8.9, 8.9, 9.1, 9.1]), line_type = '--'),
                           TimeSeries('fast_support', dates = md_dates, values = np.array([8.9, 9.0, 9.1, 9.2]), line_type = '--'),
                           TimeSeries('slow_resistance', dates = md_dates, values = np.array([9.2, 9.2, 9.4, 9.4]), line_type = '--'),
                           TimeSeries('fast_resistance', dates = md_dates, values = np.array([9.2, 9.3, 9.4, 9.5]), line_type = '--'), 
                           OHLC('price', dates = md_dates, 
                                o = np.array([8.9, 9.1, 9.3, 8.6]),
                                h = np.array([9.0, 9.3, 9.4, 8.7]),
                                l = np.array([8.8, 9.0, 9.2, 8.4]),
                                c = np.array([8.95, 9.2, 9.35, 8.5]),
                                v = np.array([200, 100, 150, 300]))
                          ] + get_long_short_trade_sets(trades, positions), title = "Price", height_ratio = 0.5)
    sig_subplot = Subplot(TimeSeries('trend', dates = md_dates, values = np.array([1, 1, -1, -1])), height_ratio=0.125, title = 'Trend')
    equity_subplot = Subplot(TimeSeries('equity', dates= pnl_dates, values = [1.0e6, 1.1e6, 1.2e6, 1.3e6]), height_ratio = 0.125, title = 'Equity', date_lines = 
                            [DateLine(date = np.datetime64('2018-01-09 14:00:00'), name = 'drawdown', color = 'red'),
                             DateLine(date = np.datetime64('2018-01-10 15:00:00'), color = 'red')], horizontal_lines = [HorizontalLine(y = 0, name = 'zero', color = 'green')])
    pos_subplot = Subplot(TimeSeries('position', dates = positions[0], values = positions[1], plot_type = 'filled_line'), height_ratio = 0.125, title = 'Position')
    annual_returns_subplot = Subplot(BucketedValues('annual returns', ['2017', '2018'], 
                                                    bucket_values = [np.random.normal(0, 1, size=(250,)), np.random.normal(0, 1, size=(500,))]),
                                                   height_ratio = 0.125, title = 'Annual Returns')
    subplot_list = [ind_subplot, sig_subplot, pos_subplot, equity_subplot, annual_returns_subplot]
    plot = Plot(subplot_list, figsize = (20,15), title = 'Test')
    plot.draw()
    
    
if __name__ == "__main__":
    test_plot()

