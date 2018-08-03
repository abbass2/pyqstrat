
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.cm as cm
from scipy.interpolate import griddata
import os
import sys
import concurrent
from pyqstrat.pq_utils import set_defaults

set_defaults()

class Experiment:
    def __init__(self, suggestion, total_cost, partial_costs):
        self.suggestion = suggestion
        self.total_cost = total_cost
        self.partial_costs = partial_costs
        
    def valid(self):
        if not all(np.isfinite(list(self.suggestion.values()))): return False
        if not np.isfinite(self.total_cost): return False
        if not all(np.isfinite(list(self.partial_costs.values()))): return False
        return True
    
    def __repr__(self):
        return f'suggestion: {self.suggestion} total cost: {self.total_cost} partial costs: {self.partial_costs}'

class Optimizer:
    def __init__(self, name, generator, cost_func, max_processes = None):
        self.name = name
        self.generator = generator
        self.cost_func = cost_func
        self.max_processes = max_processes
        self.experiments = []
        
        
    def _run_single_process(self):
        for suggestion in self.generator:
            total_cost, partial_costs = self.cost_func(suggestion)
            self.generator.send((total_cost, partial_costs))
            self.experiments.append(Experiment(suggestion, total_cost, partial_costs))
    
    #TODO: Needs to be rewritten to send costs back to generator when we do parrallel gradient descent, etc.
    def _run_multi_process(self, raise_on_error):
        fut_map = {}
        with concurrent.futures.ProcessPoolExecutor(self.max_processes) as executor:
            for suggestion in self.generator:
                future = executor.submit(self.cost_func, suggestion)
                fut_map[future] = suggestion
                
            for future in concurrent.futures.as_completed(fut_map):
                try:
                    total_cost, partial_costs = future.result()
                except Exception as e:
                    new_exc = type(e)(f'Exception: {str(e)} with suggestion: {suggestion}').with_traceback(sys.exc_info()[2])
                    if raise_on_error: raise new_exc
                    else: print(str(new_exc))
                    continue
                suggestion = fut_map[future]
                self.experiments.append(Experiment(suggestion, total_cost, partial_costs))
                #print(f'completed suggestion: {suggestion} result: {total_cost} {partial_costs}')
    
    def run(self, raise_on_error = False):
        if self.max_processes == 1: self._run_single_process()
        else: self._run_multi_process(raise_on_error)
        
    def experiment_list(self, sort_order = 'lowest_cost'):
        if sort_order == 'lowest_cost':
            experiments = sorted(self.experiments, key = lambda x : x.total_cost, reverse = True)
        elif sort_order == 'highest_cost':
            experiments = sorted(self.experiments, key = lambda x : x.total_cost, reverse = False)
        elif sort_order == 'sequence': # in order in which experiment was run
            experiments = self.experiments
        else:
            raise Exception(f'invalid sort order: {sort}')
        return experiments
    
    def plot(self, xname, yname, zname = 'all', plot_type = 'surface', figsize = (15,8), interpolation = 'linear', 
             cmap = 'viridis', marker_size = 100, marker_color = 'r', xlim = None, ylim = None):
        if len(self.experiments) == 0: return

        # Get rid of nans since matplotlib does not like them
        experiments = [experiment for experiment in self.experiments if experiment.valid()]
        if xlim:
            experiments = [experiment for experiment in experiments if experiment.suggestion[xname] >= xlim[0] and experiment.suggestion[xname] <= xlim[1]]
        if ylim:
            experiments = [experiment for experiment in experiments if experiment.suggestion[yname] >= ylim[0] and experiment.suggestion[yname] <= ylim[1]]

        xvalues = [experiment.suggestion[xname] for experiment in experiments]
        yvalues = [experiment.suggestion[yname] for experiment in experiments]
        zvalues = []

        if zname == 'all':
            zvalues.append(('cost', np.array([experiment.total_cost for experiment in experiments])))
            partial_cost_keys = experiments[0].partial_costs.keys()
            for key in partial_cost_keys:
                zvalues.append((key, [experiment.partial_costs[key] for experiment in experiments]))
        elif zname == 'cost':
            zvalues.append(('cost', np.array([experiment.total_cost for experiment in experiments])))
        else:
            zvalues.append((zname, [experiment.partial_costs[zname] for experiment in experiments]))

        subplot_kw = None
        if plot_type == 'surface':
            subplot_kw = dict(projection='3d')

        fig, axes = plt.subplots(nrows = len(zvalues), ncols = 1, figsize = figsize, squeeze = False, subplot_kw = subplot_kw)
        for i, tup in enumerate(zvalues):
            zlabel = tup[0]
            values = tup[1]
            ax = axes[i][0]
            ax.set_xlabel(xname)
            ax.set_ylabel(yname)
            xi = np.linspace(min(xvalues), max(xvalues))
            yi = np.linspace(min(yvalues), max(yvalues))
            X, Y = np.meshgrid(xi, yi)
            Z = griddata((xvalues, yvalues), values, (xi[None,:], yi[:,None]), method='linear')
            Z = np.nan_to_num(Z)
            ax.set_title(zlabel)

            if plot_type == 'surface':
                ax.set_zlabel(zlabel);
                ax.plot_surface(X, Y, Z, cmap=cmap)
                ax.scatter(xvalues, yvalues, values, marker = 'o', s = marker_size, c = marker_color)
            elif plot_type == 'contour':
                cs = ax.contour(X, Y, Z, linewidths=0.5, colors='k')
                ax.clabel(cs, cs.levels[::2], fmt = "%.3g", inline=1)
                ax.contourf(X, Y, Z, cmap = cmap)
                ax.scatter(xvalues, yvalues, marker='o', s= marker_size, c = marker_color, zorder=10)
            else:
                raise Exception(f'unknown plot type: {plot_type}')

            m = cm.ScalarMappable(cmap=cmap)
            m.set_array(Z)
            plt.colorbar(m, ax = ax)
            
def test_optimize():
    def generator():
        for x in range(0,5):
            for y in range(0,5):
                costs = (yield {'x' : x, 'y' : y})
                yield
                #print(f'got costs: {costs} with suggestion: {x} {y}')

    def cost_func(suggestion):
        x = suggestion['x']
        y = suggestion['y']
        #if x == 2 and y == 1: raise Exception('big error')
        cost = x ** 2 + y ** 2
        cost = np.sin(np.sqrt(x ** 2 + y ** 2))
        return cost, {'sharpe' : cost, 'std' : -0.1 * cost}

    optimizer = Optimizer('test', generator(), cost_func, max_processes = 1)
    optimizer.run()
    optimizer.plot(xname = 'x', yname = 'y', plot_type = 'contour', figsize = (20,20), xlim = (0,2), ylim = (0,2))
    
            
if __name__ == "__main__":
    test_optimize()

