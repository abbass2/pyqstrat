# $$_ Lines starting with # $$_* autogenerated by jup_mini. Do not modify these
# $$_code
# $$_ %%checkall
from __future__ import annotations
import numpy as np
import pandas as pd
import os
import sys
import itertools
import concurrent
import concurrent.futures
import multiprocessing as mp
from pyqstrat.pq_utils import has_display, get_child_logger
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from typing import Any, Callable, Generator
from collections.abc import Sequence

_logger = get_child_logger(__name__)


def flatten_keys(experiments: Sequence[Any]) -> list[str]:
    '''
    Utility function so we can find all keys for other costs in all experiments even if the first experiment
    does not have all of them
    '''
    keys = set()
    for exp in experiments:
        keys.update(list(exp.other_costs.keys()))
    return list(keys)


class Experiment:
    '''An Experiment stores a suggestion and its result'''
    def __init__(self, suggestion: dict[str, Any], cost: float, other_costs: dict[str, float]) -> None:
        '''
        Args:
            suggestion: A dictionary of variable name -> value
            cost: A float representing output of the function we are testing with this suggestion as input.
            other_costs: A dictionary of other results we want to store and look at later.

        '''
        self.suggestion = suggestion
        self.cost = cost
        self.other_costs = other_costs
        
    def valid(self) -> bool:
        '''
        Returns True if all suggestions and costs are finite, i.e not NaN or +/- Infinity
        '''
        if not all(np.isfinite(list(self.suggestion.values()))): return False
        if not np.isfinite(self.cost): return False
        if not all(np.isfinite(list(self.other_costs.values()))): return False
        return True
    
    def __repr__(self) -> str:
        return f'suggestion: {self.suggestion} cost: {self.cost} other costs: {self.other_costs}'


class Optimizer:
    '''Optimizer is used to optimize parameters for a strategy.'''
    def __init__(self, name: str, 
                 generator: Generator[dict[str, Any], tuple[float, dict[str, float]], None], 
                 cost_func: Callable[[dict[str, Any]], tuple[float, dict[str, float]]], 
                 max_processes: int | None = None) -> None:
        '''
        Args:
            name: Display title for plotting, etc.
            generator: A generator (see Python Generators) that takes no inputs and yields a dictionary with parameter name -> parameter value.
            cost_func: A function that takes a dictionary of parameter name -> parameter value as input and outputs cost for that set of parameters.
            max_processes: If not set, the Optimizer will look at the number of CPU cores on your machine to figure out how many processes to run.
        '''
        self.name = name
        self.generator = generator
        self.cost_func = cost_func
        import sys
        if sys.platform in ['win32', 'cygwin']:
            if max_processes is not None and max_processes != 1:
                raise Exception("max_processes must be 1 on Microsoft Windows")
            max_processes = 1
        self.max_processes = max_processes
        self.experiments: list[Experiment] = []
        
    def _run_single_process(self) -> None:
        try:
            for suggestion in self.generator:
                if suggestion is None: continue
                cost, other_costs = self.cost_func(suggestion)
                self.generator.send((cost, other_costs))
                self.experiments.append(Experiment(suggestion, cost, other_costs))
        except StopIteration:
            # Exhausted generator
            return
    
    # TODO: Needs to be rewritten to send costs back to generator when we do parallel gradient descent, etc.
    def _run_multi_process(self, raise_on_error: bool) -> None:
        fut_map = {}
        
        # on mac m1 the default start method is set to spawn so change to fork instead
        with concurrent.futures.ProcessPoolExecutor(self.max_processes, mp_context=mp.get_context('fork')) as executor:
            for suggestion in self.generator:
                if suggestion is None: continue
                future = executor.submit(self.cost_func, suggestion)
                fut_map[future] = suggestion
                
            for future in concurrent.futures.as_completed(fut_map):
                try:
                    cost, other_costs = future.result()
                except Exception as e:
                    _suggestion = fut_map.get(future)
                    new_exc = type(e)(f'Exception: {str(e)} with suggestion: {_suggestion}').with_traceback(sys.exc_info()[2])
                    if raise_on_error: raise new_exc
                    else: print(str(new_exc))
                    continue
                suggestion = fut_map[future]
                self.experiments.append(Experiment(suggestion, cost, other_costs))
    
    def run(self, raise_on_error: bool = False) -> None:
        '''Run the optimizer.
        
        Args:
            raise_on_error: If set to True, even if we are running a multiprocess optimization, any Exceptions will bubble up and stop the Optimizer.
              This can be useful for debugging to see stack traces for Exceptions.
        '''
        if self.max_processes == 1: self._run_single_process()
        else: self._run_multi_process(raise_on_error)
        
    def experiment_list(self, sort_order: str = 'lowest_cost') -> Sequence[Experiment]:
        '''Returns the list of experiments we have run
        
        Args:
            sort_order: Can be set to lowest_cost, highest_cost or sequence.  
              If set to sequence, experiments are returned in the sequence in which they were run
        '''
        if sort_order == 'lowest_cost':
            experiments = sorted(self.experiments, key=lambda x: x.cost, reverse=True)
        elif sort_order == 'highest_cost':
            experiments = sorted(self.experiments, key=lambda x: x.cost, reverse=False)
        elif sort_order == 'sequence':  # in order in which experiment was run
            experiments = self.experiments
        else:
            raise Exception(f'invalid sort order: {sort_order}')
        return experiments
    
    def df_experiments(self, sort_column: str = 'cost', ascending: bool = True) -> pd.DataFrame:
        '''
        Returns a dataframe containing experiment data, sorted by sort_column (default "cost")
        '''
        if len(self.experiments) == 0: return None
        pc_keys = flatten_keys(self.experiments)
        # pc_keys = list(self.experiments[0].other_costs.keys())
        sugg_keys = list(self.experiments[0].suggestion.keys())
        records = [[exp.suggestion[k] for k in sugg_keys] + [exp.cost] + [exp.other_costs[k] for k in pc_keys]
                   for exp in self.experiments if exp.valid()]
        df = pd.DataFrame.from_records(records, columns=sugg_keys + ['cost'] + pc_keys)
        df = df.sort_values(by=[sort_column], ascending=ascending)
        return df
    
    def plot_3d(self, 
                x: str, 
                y: str, 
                z: str = 'all', 
                markers: bool = True,
                filter_func: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
                height: int = 1000,
                width: int = 0,
                xlim: tuple[float, float] | None = None,
                ylim: tuple[float, float] | None = None, 
                vertical_spacing: float = 0.05,
                show: bool = True) -> go.Figure:
        
        """Creates a 3D plot of the optimization output for plotting 2 parameters and costs.
        
        Args:
            x: Name of the parameter to plot on the x axis, corresponding to the same name in the generator.
            y: Name of the parameter to plot on the y axis, corresponding to the same name in the generator.
            z: Can be one of:
              "cost" 
              The name of another cost variable corresponding to the output from the cost function
              "all", which creates a subplot for cost plus all other costs
            markers: If set, we show actual datapoints on the graph
            filter_func: A function that can be used to reduce the dataset before plotting.
                For example, you may want to filter on a dimension beyond x, y, z to pick a single value
                from that dimension
             marker: Adds a marker to each point in x, y, z to show the actual data used for interpolation.  You can set this to None to turn markers off.
            vertical_spacing: Vertical space between subplots
         """
        
        if len(self.experiments) == 0: 
            _logger.warning('No experiments found')
            return go.Figure()
        if not has_display(): return go.Figure()

        # Get rid of nans
        experiments = [experiment for experiment in self.experiments if experiment.valid()]
        if filter_func: experiments = filter_func(experiments)
        if not len(experiments):
            _logger.warning('No valid experiments found')
            return go.Figure()

        if xlim:
            experiments = [experiment for experiment in experiments if experiment.suggestion[x] >= xlim[0] and experiment.suggestion[x] <= xlim[1]]
        if ylim:
            experiments = [experiment for experiment in experiments if experiment.suggestion[y] >= ylim[0] and experiment.suggestion[y] <= ylim[1]]

        xvalues = np.array([experiment.suggestion[x] for experiment in experiments])
        yvalues = np.array([experiment.suggestion[y] for experiment in experiments])
        zvalues = []

        if z == 'all':
            zvalues.append(('cost', np.array([experiment.cost for experiment in experiments])))
            if len(experiments[0].other_costs):
                other_cost_keys = experiments[0].other_costs.keys()
                for key in other_cost_keys:
                    zvalues.append((key, np.array([experiment.other_costs[key] for experiment in experiments])))
        elif z == 'cost':
            zvalues.append(('cost', np.array([experiment.cost for experiment in experiments])))
        else:
            zvalues.append((z, np.array([experiment.other_costs[z] for experiment in experiments])))

        cols: dict[str, np.ndarray] = dict(x=xvalues, y=yvalues)
        for metric, _z in zvalues:
            cols[metric] = _z
        _df = pd.DataFrame(cols).sort_values(by=['x', 'y'])
        x = np.unique(_df.x.values)
        y = np.unique(_df.y.values)
        df = _df.set_index(['x', 'y']).reindex(itertools.product(x, y)).reset_index()

        metrics = np.unique([metric[0] for metric in zvalues])
        _z = np.full((len(metrics), len(x), len(y)), np.nan)

        for i, metric in enumerate(metrics):
            _z[i, :, :] = df[metric].values.reshape((len(x), len(y)))

        fig = make_subplots(rows=len(metrics), cols=1, subplot_titles=metrics, shared_xaxes=True, vertical_spacing=vertical_spacing)
        fig.update_layout(height=height)

        num_metrics = len(metrics)
        colorbar_height = 1 / (num_metrics + 1)

        for i, metric in enumerate(metrics):
            zmatrix = _z[i]
            row = i + 1
            zero = 0 - np.nanmin(zmatrix) / (np.nanmax(zmatrix) - np.nanmin(zmatrix))
            colorscale = [
                [0, 'rgba(237, 100, 90, 0.85)'],   
                [zero, 'white'],  
                [1, 'rgba(17, 165, 21, 0.85)']]  
            colorbar_y = 1 - (i + 1) * colorbar_height
            trace = go.Contour(x=x, 
                               y=y, 
                               z=zmatrix, 
                               name=metric, 
                               colorscale=colorscale, 
                               colorbar=dict(len=colorbar_height, y=colorbar_y),
                               connectgaps=True,
                               contours=dict(showlabels=True, labelfont=dict(color='white')))
            if markers:
                scatter = go.Scatter(x=df.x, y=df.y, marker=dict(color=df[metric].values), mode='markers')
                fig.add_trace(scatter, row=row, col=1)
            fig.add_trace(trace, row=row, col=1)

        fig.update_layout(showlegend=False)
        if show: fig.show()
        return fig

    def plot_2d(self, 
                x: str, 
                y: str = 'all',
                title: str = '',
                marker_mode: str = 'lines+markers', 
                height: int = 1000,
                width: int = 0,
                show: bool = True) -> go.Figure:
        """Creates a 2D plot of the optimization output for plotting 1 parameter and costs
        
        Args:
            x: Name of the parameter to plot on the x axis, corresponding to the same name in the generator.
            y: Can be one of:
              "cost" 
              The name of another cost variable corresponding to the output from the cost function
              "all", which creates a subplot for cost plus all other costs
            marker_mode: see plotly mode.  Set to 'lines' to turn markers off
         """
        if len(self.experiments) == 0:
            _logger.warning('No experiments found')
            return
        if not has_display(): return go.Figure()

        # Get rid of nans
        experiments = [experiment for experiment in self.experiments if experiment.valid()]

        xvalues = [experiment.suggestion[x] for experiment in experiments]
        yvalues = []

        if y == 'all':
            yvalues.append(('cost', np.array([experiment.cost for experiment in experiments])))
            other_cost_keys = experiments[0].other_costs.keys()
            for key in other_cost_keys:
                yvalues.append((key, np.array([experiment.other_costs[key] for experiment in experiments])))
        elif y == 'cost':
            yvalues.append(('cost', np.array([experiment.cost for experiment in experiments])))
        else:
            yvalues.append((y, np.array([experiment.other_costs[y] for experiment in experiments])))

        xarray = np.array(xvalues)
        x_sort_indices = np.argsort(xarray)
        xarray = xarray[x_sort_indices]
        fig = make_subplots(rows=len(yvalues), cols=1)

        for i, tup in enumerate(yvalues):
            name = tup[0]
            yarray = tup[1]
            yarray = yarray[x_sort_indices]
            trace = go.Scatter(name=name, x=xarray, y=yarray, mode=marker_mode)
            row = i + 1
            fig.add_trace(trace, row=row, col=1)
            fig.update_xaxes(title_text=x, row=row, col=1)
            fig.update_yaxes(title_text=name, row=row, col=1)

        fig.update_layout(height=height, title=title, showlegend=False)

        if show: fig.show()
        return fig


# Functions used in unit testing
def _generator_1d() -> Generator[dict[str, Any], tuple[float, dict[str, float]], None]:
    for x in np.arange(0, np.pi * 2, 0.1):
        _ = (yield {'x': x})


def _cost_func_1d(suggestion: dict[str, Any]) -> tuple[float, dict[str, float]]:
    x = suggestion['x']
    cost = np.sin(x)
    ret = (cost, {'std': -0.1 * cost})
    return ret


def _generator_2d() -> Generator[dict[str, Any], tuple[float, dict[str, float]], None]:
    for x in np.arange(0, np.pi * 2, 0.5):
        for y in np.arange(0, np.pi * 2, 0.5):
            _ = (yield {'x': x, 'y': y})


def _cost_func_2d(suggestion: dict[str, Any]) -> tuple[float, dict[str, float]]:
    x = suggestion['x']
    y = suggestion['y']
    cost = np.sin(np.sqrt(x**2 + y ** 2))
    return cost, {'sharpe': cost, 'std': -0.1 * cost}

            
def test_optimize():
    max_processes = 1 if os.name == 'nt' else 4

    optimizer_1d = Optimizer('test', _generator_1d(), _cost_func_1d, max_processes=max_processes)
    optimizer_1d.run(raise_on_error=True)
    optimizer_1d.plot_2d(x='x', marker_mode='lines+markers', title='Optimizer 1D Test')
    
    optimizer_2d = Optimizer('test', _generator_2d(), _cost_func_2d, max_processes=max_processes)
    optimizer_2d.run()
    optimizer_2d.plot_3d(x='x', y='y')
            

if __name__ == "__main__":
    test_optimize()
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
# $$_end_code
