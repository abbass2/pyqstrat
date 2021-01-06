# Library Imports
import os
import sys
import math
import unittest
import doctest
import pandas as pd
import numpy as np
from dataclasses import dataclass
from IPython.core.display import display
from IPython.core.display import clear_output
from ipywidgets import widgets
import plotly
import plotly.callbacks
import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots

from typing import List, Tuple, Callable, Any, Sequence, Dict, Optional
import traitlets
import pyqstrat as pq


# Local Imports
ROOT_DIR = os.path.join(sys.path[0])
sys.path.insert(1, ROOT_DIR)

# Constants and Globals
_paths = pq.get_paths('..')
_calendar = pq.Calendar.get_calendar(pq.Calendar.NYSE)
_logger = pq.get_child_logger(__name__)

LineDataType = Tuple[str, 
                     pd.DataFrame, 
                     Dict[Any, pd.DataFrame]]
    
DimensionFilterType = Callable[[
    pd.DataFrame,
    str,
    List[Tuple[str, Any]]],
    np.ndarray]

DataFilterType = Callable[[
    pd.DataFrame,
    List[Tuple[str, Any]]],
    pd.DataFrame]

StatFuncType = Callable[[pd.DataFrame, str, str, str], List[LineDataType]]

DetailDisplayType = Callable[[
    widgets.Widget, 
    pd.DataFrame],
    None]


PlotFuncType = Callable[[str, str, List[LineDataType]], List[widgets.Widget]]

DataFrameTransformFuncType = Callable[[pd.DataFrame], pd.DataFrame]

SeriesTransformFuncType = Callable[[pd.Series], pd.Series]

DisplayFormFuncType = Callable[[Sequence[widgets.Widget]], None]

UpdateFormFuncType = Callable[[int], None]

CreateSelectionWidgetsFunctype = Callable[[Dict[str, str], Dict[str, str], UpdateFormFuncType], Dict[str, Any]]


def display_form(form_widgets: Sequence[widgets.Widget]) -> None:
    clear_output()
    box_layout = widgets.Layout(
        display='flex',
        flex_flow='column',
        align_items='stretch',
        border='solid',
        width='100%')
    box = widgets.Box(children=list(form_widgets), layout=box_layout)
    display(box)
    
    
class SimpleTransform:
    def __init__(self, transforms: List[Tuple[str, str, SeriesTransformFuncType]] = None) -> None:
        self.transforms = [] if transforms is None else transforms
        
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        for (colname, label, func) in self.transforms:
            data[label] = func(data[colname])
        return data
    

def simple_dimension_filter(data: pd.DataFrame, dim_name: str, selected_values: List[Tuple[str, Any]]) -> np.ndarray:
    '''
    Produces a list to put into a dropdown for selecting a dimension value
    '''
    mask = np.full(len(data), True)
    for name, value in selected_values:
        mask &= (data[name] == value)
    return np.unique(data[mask][dim_name].values)  # will sort values before returning them
    
    
def simple_data_filter(data: pd.DataFrame, selected_values: List[Tuple[str, Any]]) -> pd.DataFrame:
    '''
    Filters a dataframe based on the selected values
    '''
    mask = np.full(len(data), True)
    for name, value in selected_values:
        mask &= (data[name] == value)
    return data[mask]
    
    
class MeanWithCI:
    '''
    Computes mean (or median) and optionally confidence intervals for plotting
    '''
    def __init__(self, mean: bool = True, ci_level: int = 0) -> None:
        '''
        Args:
            mean: If set, compute mean, otherwise compute median. Default True
            ci_level: Set to 0 for no confidence intervals, or the level you want.
                For example, set to 95 to compute a 95% confidence interval. Default 0
        '''
        self.mean = mean
        self.ci_level = ci_level
        
    def __call__(self, filtered_data: pd.DataFrame, xcol: str, ycol: str, zcol: str) -> List[LineDataType]:
        '''
        For each unique value of x and z, compute mean (and optionally ci) of y.
        Return:
            x, y data for plotting lines of the mean of y versus x for each z and the data used to compute the mean
        '''
        zvals = np.unique(filtered_data[zcol])
        cols = [col for col in filtered_data.columns if col not in [xcol, ycol, zcol]]
        df = filtered_data[[xcol, ycol, zcol] + cols]
        ret = []
        
        for zvalue in zvals:
            df = filtered_data[filtered_data[zcol] == zvalue]
            line = df[[xcol, ycol]].groupby(xcol, as_index=False)[[ycol]].mean()
            line = line[[xcol, ycol]]
            ret.append((zvalue, line, df))
        return ret
    
    
class SimpleDetailTable:
    '''
    Displays a pandas DataFrame under a plot that contains the data used to compute a statistic of y for each x, y pair
    '''
    
    def __init__(self, colnames: Optional[List[str]] = None, float_format: str = '{:.4g}', min_rows: int = 100) -> None:
        '''
        Args:
            colnames: List of column names to display. If None we display all columns. Default None
            float_format: Format for each floating point column. Default {:.4g}
            min_rows: Do not truncate the display of the table before this many rows. Default 100
       '''
        self.colnames = colnames
        self.float_format = float_format
        self.min_rows = min_rows
        
    def __call__(self, detail_widget: widgets.Widget, data: pd.DataFrame) -> None:
        '''
        Args:
            detail_widget: The widget to display the data in
            data: The dataframe to display
        '''
        if self.float_format:
            orig_float_format = pd.options.display.float_format
            pd.options.display.float_format = (self.float_format).format
        
        if self.min_rows:
            orig_min_rows = pd.options.display.min_rows
            pd.options.display.min_rows = self.min_rows
            
        with detail_widget:
            clear_output()
            if self.colnames: data = data[self.colnames]
            display(data.reset_index(drop=True))
                
        if self.float_format: pd.options.display.float_format = orig_float_format
        if self.min_rows: pd.options.display.min_rows = orig_min_rows
            
            
def create_selection_dropdowns(dims: Dict[str, str], labels: Dict[str, str], update_form_func: UpdateFormFuncType) -> Dict[str, Any]:
    '''
    Create a list of selection widgets
    '''
    selection_widgets: Dict[str, widgets.Widget] = {}
    for name in dims.keys():
        label = labels[name] if name in labels else name
        widget = widgets.Dropdown(description=label, style={'description_width': 'initial'})
        selection_widgets[name] = widget
        
    for widget in selection_widgets.values():
        widget.observe(lambda x: on_widgets_updated(x, update_form_func, selection_widgets), names='value')

    return selection_widgets


def on_widgets_updated(change: traitlets.utils.bunch.Bunch, update_form_func, selection_widgets: Dict[str, widgets.Widget]) -> None:
    '''
    Callback called by plotly when widgets are updated by the user.
    '''
    owner = change['owner']
    widgets = list(selection_widgets.values())
    owner_idx = widgets.index(owner)
    update_form_func(owner_idx)
            
            
@dataclass
class LineConfig:
    color: Optional[str] = None
    thickness: float = math.nan
    secondary_y: bool = False
    marker_mode: str = 'lines+markers'
    show_detail: bool = True
        
        
class LineGraphWithDetailDisplay:
    '''
    Draws line graphs and also includes a detail pane.
    When you click on a point on the line graph, the detail pane shows the data used to compute that point.
    '''
    
    def __init__(self, 
                 display_detail_func: DetailDisplayType = SimpleDetailTable(), 
                 line_configs: Dict[str, LineConfig] = {}, 
                 title: str = None, 
                 hovertemplate: str = None) -> None:
        '''
        Args:
            display_detail_func: A function that displays the data on the detail pane. Default SimpleDetailTable
            line_configs: Configuration of each line. The key in this dict is the zvalue for that line.  Default {}
            title: Title of the graph. Default None
            hovertemplate: What to display when we hover over a point on the graph.  See plotly hovertemplate
        '''
        self.display_detail_func = display_detail_func
        self.line_configs = line_configs
        self.title = title
        self.hovertemplate = hovertemplate
        self.default_line_config = LineConfig()
        self.detail_data: Dict[Any, pd.DataFrame] = {}
        self.xcol = ''
 
    def __call__(self, xaxis_title: str, yaxis_title: str, line_data: List[LineDataType]) -> List[widgets.Widget]:
        '''
        Draw the plot and also set it up so if you click on a point, we display the data used to compute that point.
        Args:
            line_data: The zvalue, plot data, and detail data for each line to draw. The plot data must have 
                x as the first column and y as the second column
         Return:
            A list of widgets to draw.  In this case, a figure widget and a output widget which contains the detail display
        '''
        if not len(line_data): return []
        self.detail_data.clear()
        secondary_y = any([lc.secondary_y for lc in self.line_configs.values()])
        
        fig_widget = go.FigureWidget(make_subplots(specs=[[{"secondary_y": secondary_y}]]))
        detail_widget = widgets.Output()

        for line_num, (zvalue, line_df, _detail_data) in enumerate(line_data):
            x = line_df.iloc[:, 0].values
            self.xcol = line_df.columns[0]
            y = line_df.iloc[:, 1].values
            self.detail_data[zvalue] = _detail_data
            line_config = self.line_configs[zvalue] if zvalue in self.line_configs else self.default_line_config
            marker_mode = line_config.marker_mode
            color = line_config.color if line_config.color else DEFAULT_PLOTLY_COLORS[line_num]
            
            hovertemplate = self.hovertemplate
            
            if hovertemplate is None:
                hovertemplate = f'Series: {zvalue} {xaxis_title}: ' + '%{x} ' + f'{yaxis_title}: ' + '%{y:.2f}'
                
            trace = go.Scatter(
                x=x,
                y=y,
                mode=marker_mode,
                name=zvalue,
                line=dict(color=color),
                hovertemplate=hovertemplate              
            )
            fig_widget.add_trace(trace, secondary_y=line_config.secondary_y)
            
            if line_config.show_detail:
                fig_widget.data[line_num].on_click(self._on_graph_click, append=True)
            
        fig_widget.update_layout(title=self.title, xaxis_title=xaxis_title)
        
        if secondary_y:
            fig_widget.update_yaxes(title_text=yaxis_title, secondary_y=True)
        else:
            fig_widget.update_layout(yaxis_title=yaxis_title)
            
        self.fig_widget = fig_widget
        self.detail_widget = detail_widget
        self.line_data = line_data
                        
        return [self.fig_widget, self.detail_widget]
    
    def _on_graph_click(self, 
                        trace: go.Trace, 
                        points: plotly.callbacks.Points, 
                        selector: plotly.callbacks.InputDeviceState) -> None:
        '''
        Callback called by plotly when you click on a point on the graph.
        When you click on a point, we display the dataframe with the data we used to compute that point.
        '''
        if not len(points.xs): return
        trace_idx = points.trace_index
        # trace_idx = trace_idx // 2  # we have two traces for each z
        zvalue = list(self.detail_data.keys())[trace_idx]
        _detail_data = self.detail_data[zvalue]
        mask = np.full(len(_detail_data), True)
        x_data = _detail_data[self.xcol].values
        mask &= (x_data == points.xs[0])
        _detail_data = _detail_data[mask]
        self.display_detail_func(self.detail_widget, _detail_data)


class InteractivePlot:
    '''
    Creates a multidimensional interactive plot off a dataframe.
    '''
    def __init__(self,
                 data: pd.DataFrame,
                 labels: Dict[str, str] = None,
                 transform_func: DataFrameTransformFuncType = SimpleTransform(),
                 create_selection_widgets_func: CreateSelectionWidgetsFunctype = create_selection_dropdowns,
                 dim_filter_func: DimensionFilterType = simple_dimension_filter,
                 data_filter_func: DataFilterType = simple_data_filter,
                 stat_func: StatFuncType = MeanWithCI(),
                 plot_func: PlotFuncType = LineGraphWithDetailDisplay(),
                 display_form_func: DisplayFormFuncType = display_form) -> None:
        '''
        Args:
            data: The pandas dataframe to use for plotting
            labels: A dict where column names from the dataframe are mapped to user friendly labels. For any column names
                not found as keys in this dict, we use the column name as the label. Default None
            dim_filter_func: A function that generates the values of a dimension based on other dimensions. For example, if 
                the user chooses "Put Option" in a put/call dropdown, the valid strikes could change in a Strike 
                dropdown that follows. Default simple_dimension_filter
            data_filter_func: A function that filters the data to plot. For example, if the user chooses "Put Option" in a put/call dropdown,
                we could filter the dataframe to only include put options. Default simple_data_filter
            stat_func: Once we have filtered the data, we may need to plot some statistics, such as mean and confidence intervals.
                In this function, we compute these statistics. Default MeanWithCI()
            plot_func: A function that plots the data.  This could also display detail data used to compute the statistics associated
                with each data point.
            display_form_func: A function that displays the form given a list of plotly widgets (including the graph widget)
        '''
        self.data = transform_func(data)
        self.create_selection_widgets_func = create_selection_widgets_func
        if labels is None: labels = {}
        self.labels = labels
        self.dim_filter_func = dim_filter_func
        self.data_filter_func = data_filter_func
        self.stat_func = stat_func
        self.plot_func = plot_func
        self.display_form_func = display_form_func
        self.selection_widgets: Dict[str, Any] = {}
        
    def create_pivot(self, xcol: str, ycol: str, zcol: str, dimensions: Dict[str, Any]) -> None:
        '''
        Create the initial pivot
        Args:
            xcol: Column name to use as the x axis in the DataFrame
            ycol: Column name to use as the y axis in the DataFrame
            zcol: Column name to use for z-values. Each zvalue can be used for a different trace within this plot. For example, a column
                called "option_type" could contain the values "American", "European", "Bermudan" and we could plot the data for each type
                in a separate trace
            dimensions: The column names used for filter dimensions. For example, we may want to filter by days to expiration and put/call
                The key the column name and the value is the initial value for that column. For example, in a 
                dropdown for Put/Call we may want "Put" to be the initial value set in the dropdown.  Set to None if you 
                don't care what initial value is chosen.
        '''
        self.xlabel = xcol if xcol not in self.labels else self.labels[xcol]
        self.ylabel = ycol if ycol not in self.labels else self.labels[ycol]
        self.zcol = zcol
        self.xcol = xcol
        self.ycol = ycol
        self.selection_widgets = self.create_selection_widgets_func(dimensions, self.labels, self.update)
        self.update()
        
    def update(self, owner_idx: int = -1) -> None:
        '''
        Redraw the form using the values of all widgets above and including the one with index owner_idx.
        If owner_idx is -1, we redraw everything.
        '''
        select_conditions = [(name, widget.value) for name, widget in self.selection_widgets.items()]
        if owner_idx == -1:
            dim_select_conditions = []
        else:
            dim_select_conditions = select_conditions[:owner_idx + 1]  # for selecting lower widget options, use value of widgets above 
        
        for name in list(self.selection_widgets.keys())[owner_idx + 1:]:
            widget = self.selection_widgets[name]
            widget.options = self.dim_filter_func(self.data, name, dim_select_conditions)
            
        if owner_idx == -1: return
                         
        filtered_data = self.data_filter_func(self.data, select_conditions)
        lines = self.stat_func(filtered_data, self.xcol, self.ycol, self.zcol)
        plot_widgets = self.plot_func(self.xlabel, self.ylabel, lines)
        self.display_form_func(list(self.selection_widgets.values()) + plot_widgets)

        
# Unit Tests
class TestInteractivePlot(unittest.TestCase):
    def test_1(self):
        np.random.seed(0)
        size = 10000
        dte = np.random.randint(5, 10, size)
        put_call = np.random.choice(['put', 'call'], size)
        year = np.random.choice([2018, 2019, 2020, 2021], size)
        delta = np.random.uniform(0, 0.5, size)
        delta = np.where(put_call == 'call', delta, -delta)
        premium = np.abs(delta * 10) * dte + np.random.normal(size=size) * dte / 10
        data = pd.DataFrame({'dte': dte, 'put_call': put_call, 'year': year, 'delta': delta, 'premium': premium})
        labels = {'premium': 'Premium $', 'year': 'Year', 'dte': 'Days to Expiry', 'delta_rnd': 'Delta'}
        ip = InteractivePlot(data, 
                             labels, 
                             transform_func=self.transform)
        ip.create_pivot('delta_rnd', 'premium', 'put_call', dimensions={'year': 2018, 'dte': None})
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data['delta_rnd'] = np.abs(pq.np_bucket(data.delta, np.arange(-0.5, 0.6, 0.1)))
        return data
    
      
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
    print('done')
