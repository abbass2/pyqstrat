
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.cm as cm
from scipy.interpolate import griddata
import os
import sys
import concurrent

from pyqstrat.pq_utils import set_defaults, has_display
from pyqstrat.plot import *

set_defaults()

class Experiment:
    '''An Experiment stores a suggestion and its result
    
    Attributes:
        suggestion: A dictionary of variable name -> value
        cost: A float representing output of the function we are testing with this suggestion as input.
        other_costs: A dictionary of other results we want to store and look at later.
    '''
    def __init__(self, suggestion, cost, other_costs):
        self.suggestion = suggestion
        self.cost = cost
        self.other_costs = other_costs
        
    def valid(self):
        '''
        Returns True if all suggestions and costs are finite, i.e not NaN or +/- Infinity
        '''
        if not all(np.isfinite(list(self.suggestion.values()))): return False
        if not np.isfinite(self.cost): return False
        if not all(np.isfinite(list(self.other_costs.values()))): return False
        return True
    
    def __repr__(self):
        return f'suggestion: {self.suggestion} cost: {self.cost} other costs: {self.other_costs}'

class Optimizer:
    '''Optimizer is used to optimize parameters for a strategy.'''
    def __init__(self, name, generator, cost_func, max_processes = None):
        '''
        Args:
            name: string used to display title in plotting, etc.
            generator: A generator (see Python Generators) that takes no inputs and yields a list of dictionaries with parameter name -> parameter value.
            cost_func: A function that takes a dictionary of parameter name -> parameter value as input and outputs cost for that set of parameters.
            max_processes: If not set, the Optimizer will look at the number of CPU cores on your machine to figure out how many processes to run.
        '''
        self.name = name
        self.generator = generator
        self.cost_func = cost_func
        self.max_processes = max_processes
        self.experiments = []
        
        
    def _run_single_process(self):
        for suggestion in self.generator:
            if suggestion is None: continue
            cost, other_costs = self.cost_func(suggestion)
            self.generator.send((cost, other_costs))
            self.experiments.append(Experiment(suggestion, cost, other_costs))
    
    #TODO: Needs to be rewritten to send costs back to generator when we do parrallel gradient descent, etc.
    def _run_multi_process(self, raise_on_error):
        fut_map = {}
        with concurrent.futures.ProcessPoolExecutor(self.max_processes) as executor:
            for suggestion in self.generator:
                if suggestion is None: continue
                future = executor.submit(self.cost_func, suggestion)
                fut_map[future] = suggestion
                
            for future in concurrent.futures.as_completed(fut_map):
                try:
                    cost, other_costs = future.result()
                except Exception as e:
                    new_exc = type(e)(f'Exception: {str(e)} with suggestion: {suggestion}').with_traceback(sys.exc_info()[2])
                    if raise_on_error: raise new_exc
                    else: print(str(new_exc))
                    continue
                suggestion = fut_map[future]
                self.experiments.append(Experiment(suggestion, cost, other_costs))
                #print(f'completed suggestion: {suggestion} result: {cost} {other_costs}')
                
    
    def run(self, raise_on_error = False):
        '''Run the optimizer.
        
        Args:
            raise_on_error: If set to True, even if we are running a multiprocess optimization, any Exceptions will bubble up and stop the Optimizer.
              This can be useful for debugging to see stack traces for Exceptions.
        '''
        if self.max_processes == 1: self._run_single_process()
        else: self._run_multi_process(raise_on_error)
        
    def experiment_list(self, sort_order = 'lowest_cost'):
        '''Returns the list of experiments we have run
        
        Args:
            sort_order: Can be set to lowest_cost, highest_cost or sequence.  
              If set to sequence, experiments are returned in the sequence in which they were run
        '''
        if sort_order == 'lowest_cost':
            experiments = sorted(self.experiments, key = lambda x : x.cost, reverse = True)
        elif sort_order == 'highest_cost':
            experiments = sorted(self.experiments, key = lambda x : x.cost, reverse = False)
        elif sort_order == 'sequence': # in order in which experiment was run
            experiments = self.experiments
        else:
            raise Exception(f'invalid sort order: {sort}')
        return experiments
    
    def df_experiments(self, sort_column = 'cost', ascending = True):
        '''
        Returns a dataframe containing experiment data, sorted by sort_column (default "cost")
        '''
        if len(self.experiments) == 0: return None
        pc_keys = list(self.experiments[0].other_costs.keys())
        sugg_keys = list(self.experiments[0].suggestion.keys())
        records = [[exp.suggestion[k] for k in sugg_keys] + [exp.cost] + [exp.other_costs[k] for k in pc_keys] for exp in self.experiments]
        df = pd.DataFrame.from_records(records, columns = sugg_keys + ['cost'] + pc_keys)
        df.sort_values(by = [sort_column], ascending = ascending, inplace = True)
        return df
    
    def plot_3d(self, x, y, z = 'all', plot_type = 'surface', figsize = (15,15), interpolation = 'linear', 
             cmap = 'viridis', marker = 'X', marker_size = 50, marker_color = 'r', xlim = None, ylim = None, hspace = None):
        
        """Creates a 3D plot of the optimization output for plotting 2 parameters and costs.
        
        Args:
            x: Name of the parameter to plot on the x axis, corresponding to the same name in the generator.
            y: Name of the parameter to plot on the y axis, corresponding to the same name in the generator.
            z: Can be one of:
              "cost" 
              The name of another cost variable corresponding to the output from the cost function
              "all", which creates a subplot for cost plus all other costs
            plot_type: surface or contour (default surface)
            figsize: Figure size
            interpolation: Can be ‘linear’, ‘nearest’ or ‘cubic’ for plotting z points between the ones passed in.  See scipy.interpolate.griddata for details
            cmap: Colormap to use (default viridis).  See matplotlib colormap for details
            marker: Adds a marker to each point in x, y, z to show the actual data used for interpolation.  You can set this to None to turn markers off.
            hspace: Vertical space between subplots
         """
        
        if len(self.experiments) == 0: return
        if not has_display(): return

        # Get rid of nans since matplotlib does not like them
        experiments = [experiment for experiment in self.experiments if experiment.valid()]
        if xlim:
            experiments = [experiment for experiment in experiments if experiment.suggestion[x] >= xlim[0] and experiment.suggestion[x] <= xlim[1]]
        if ylim:
            experiments = [experiment for experiment in experiments if experiment.suggestion[y] >= ylim[0] and experiment.suggestion[y] <= ylim[1]]

        xvalues = [experiment.suggestion[x] for experiment in experiments]
        yvalues = [experiment.suggestion[y] for experiment in experiments]
        zvalues = []

        if z == 'all':
            zvalues.append(('cost', np.array([experiment.cost for experiment in experiments])))
            other_cost_keys = experiments[0].other_costs.keys()
            for key in other_cost_keys:
                zvalues.append((key, np.array([experiment.other_costs[key] for experiment in experiments])))
        elif z == 'cost':
            zvalues.append(('cost', np.array([experiment.cost for experiment in experiments])))
        else:
            zvalues.append((z, np.array([experiment.other_costs[zname] for experiment in experiments])))
            
        subplots = []
        for tup in zvalues:
            name = tup[0]
            zarray = tup[1]
            if plot_type == 'contour':
                zlabel = None
                title = name
            else:
                zlabel = name
                title = None
                
            subplots.append(Subplot(data_list = [
                XYZData(name, xvalues, yvalues, zarray, plot_type = plot_type, 
                        marker = marker, marker_size = marker_size, marker_color = marker_color, interpolation = interpolation, cmap = cmap
                    )], title = title, xlabel = x, ylabel = y, zlabel = zlabel, xlim = xlim, ylim = ylim))
        plot = Plot(subplots, figsize = figsize, title = 'Optimizer 2D Test', hspace = hspace)
        plot.draw()
        
    def plot_2d(self, x, y = 'all', plot_type = 'line', figsize = (15,8), marker = 'X', marker_size = 50, marker_color = 'r', xlim = None, hspace = None):
        """Creates a 2D plot of the optimization output for plotting 1 parameter and costs.
        
        Args:
            x: Name of the parameter to plot on the x axis, corresponding to the same name in the generator.
            y: Can be one of:
              "cost" 
              The name of another cost variable corresponding to the output from the cost function
              "all", which creates a subplot for cost plus all other costs
            plot_type: line or scatter (default line)
            figsize: Figure size
            marker: Adds a marker to each point in x, y to show the actual data used for interpolation.  You can set this to None to turn markers off.
            hspace: Vertical space between subplots
         """
        if len(self.experiments) == 0: return
        if not has_display(): return

        # Get rid of nans since matplotlib does not like them
        experiments = [experiment for experiment in self.experiments if experiment.valid()]
        if xlim:
            experiments = [experiment for experiment in experiments if experiment.suggestion[x] >= xlim[0] and experiment.suggestion[x] <= xlim[1]]

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
            yvalues.append((y, np.array([experiment.other_costs[zname] for experiment in experiments])))
            
        subplots = []
        for tup in yvalues:
            name = tup[0]
            yarray = tup[1]
            subplots.append(Subplot(data_list = [
                XYData(name, xvalues, yarray, plot_type = plot_type, 
                        marker = marker, marker_size = marker_size, marker_color = marker_color
                    )], xlabel = x, ylabel = name, xlim = xlim))
        plot = Plot(subplots, figsize = figsize, title = 'Optimizer 1D Test')
        plot.draw()
            
def test_optimize():
    
    def generator_2d():
        for x in range(0,5):
            for y in range(0,5):
                costs = (yield {'x' : x, 'y' : y})
                yield

    def cost_func_2d(suggestion):
        x = suggestion['x']
        y = suggestion['y']
        cost = x ** 2 + y ** 2
        cost = np.sin(np.sqrt(x ** 2 + y ** 2))
        return cost, {'sharpe' : cost, 'std' : -0.1 * cost}

    optimizer_2d = Optimizer('test', generator_2d(), cost_func_2d, max_processes = 1)
    optimizer_2d.run()
    optimizer_2d.plot_3d(x = 'x', y = 'y')

    def generator_1d():
        for x in range(0,5):
            costs = (yield {'x' : x})
            yield

    def cost_func_1d(suggestion):
        x = suggestion['x']
        cost = np.sin(x)
        return cost, {'std' : -0.1 * cost}

    optimizer_1d = Optimizer('test', generator_1d(), cost_func_1d, max_processes = 1)
    optimizer_1d.run()
    optimizer_1d.plot_2d(x = 'x', plot_type = 'line', marker = 'o', marker_color = 'blue')
            
if __name__ == "__main__":
    test_optimize()

