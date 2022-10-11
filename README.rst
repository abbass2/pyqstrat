|PyVersion| |Status| |License|

Introduction
============

The ``pyqstrat`` package is designed for backtesting quantitative strategies. It was originally built for my own use as a quant trader / researcher, after I could not find a python based framework that was fast, extensible and transparent enough for use in my work.  

This framework is designed for capable programmers who are comfortable with numpy and reasonably advanced Python techniques.

The goals are:

* Speed - Performance sensitive components are written at the numpy level, or in C++, which can lead to performance improvement of several orders of magnitude over regular Python code.  Where possible, we parrallelize work so you can take advantage of all the cores available on your machine.
* Transparency - If you are going to commit money to a strategy, you want to know exactly what assumptions you are making.  The code is written and documented so these are as clear as possible.
* Extensibility - It would be impossible to think of all requirements for backtesting strategies that traders could come up with. In addition, traders will want to measure different metrics depending on the strategy being traded.

Using this framework, you can:

* Create indicators, trading signals, trading rules and market simulators and add them to a strategy
* Create contract groups for PNL grouping. For example, for futures and options, you often have a group such as the "front-month future" or the "delta hedge" where the actual instruments change over time but you still want to analyze PNL at the contract group level.
* Reuse existing market simulation or add your own assumptions to simulate how and when orders are filled
* Measure returns, drawdowns, common return metrics such as sharpe, calmar and also add your own metrics.
* Optimize your strategy's parameters using all the CPU cores on your machine.


** NOTE: This is beta software and the API will change **

Installation
------------
I would strongly recommend installing anaconda and creating an anaconda environment. I personally prefer miniconda which runs much faster than the original anaconda.  See https://github.com/conda-forge/miniforge for installation instructions.

pyqstrat relies on numpy, scipy, matplotlib and pandas which in turn use Fortran and C code that needs to be compiled.

::

   conda install --channel conda-forge libzip

   pip install pyqstrat

Requirements:

* Python_ version 3.7 or higher;

Documentation
-------------

The best way to get started is to go through this Jupyter notebook: `Building Strategies <https://github.com/abbass2/pyqstrat/tree/master/pyqstrat/notebooks/building_strategies.ipynb>`_

`Jupyter Notebooks <https://github.com/abbass2/pyqstrat/tree/master/pyqstrat/notebooks>`_ 

`API docs <https://abbass2.github.io/pyqstrat>`_

Discussion
----------

The `pyqstrat user group <https://groups.io/g/pyqstrat>`_ is the group used for pyqstrat discussions.


Acknowledgements
----------------

Before building this, I looked at the following.  Although I ended up not using them, they are definitely worth looking at.

`R quantstrat library <https://github.com/braverock/quantstrat>`_

`Python backtrader project <https://www.backtrader.com>`_


Some of the ideas I use in this framework come from the following books

`Trading Systems: A New Approach to System Development and Portfolio Optimisation - Tomasini, Emilio and Jaekle, Urban <https://www.amazon.com/gp/product/1905641796/ref=oh_aui_search_detailpage?ie=UTF8&psc=1>`_

`Machine Trading - Chan, Ernie <https://www.amazon.com/gp/product/1119219604>`_

`Algorithmic Trading: Winning Strategies and Their Rationale - Chan, Ernie <https://www.amazon.com/gp/product/1118460146>`_


Disclaimer
----------

The software is provided on the conditions of the simplified BSD license.

.. _Python: http://www.python.org

.. |PyVersion| image:: https://img.shields.io/badge/python-3.7+-blue.svg
   :alt:

.. |Status| image:: https://img.shields.io/badge/status-beta-green.svg
   :alt:

.. |License| image:: https://img.shields.io/badge/license-BSD-blue.svg
   :alt:
   
