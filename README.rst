|PyVersion| |Status| |License|

Introduction
============

The ``pyqstrat`` package is designed for backtesting quantitative strategies. It was originally built for my own use after I could not find a python based framework that was fast, extensible and transparent enough for use in my work.  

The goals are:

* Speed - Performance sensitive components are written at the numpy level, or in cython or C++, which can lead to performance gains of a couple of orders of magnitude over Python code.
* Transparency - If you are going to commit money to a strategy, you want to know exactly what assumptions it includes. The code is written and documented so these are as clear as possible.
* Extensibility - It would be impossible to think of all requirements for backtesting strategies that traders could come up with. In addition, it's important to measure custom metrics relevant to the strategy being traded.

Using this framework, you can:

* Create indicators, trading signals, trading rules and market simulators and add them to a strategy
* Create contract groups for PNL grouping. For example, for futures and options, you may create a "front-month future" and "delta hedge" where the actual instruments change over time but you still want to analyze PNL at the contract group level.
* Reuse existing market simulation or add your own assumptions to simulate when and at what price orders are filled
* Measure returns, drawdowns, common return metrics such as sharpe, calmar and also add your own metrics.
* Optimize your strategy's parameters taking advantage of all the CPUs on a machine




Installation
------------
I would strongly recommend installing mamba and creating a mamba environment. See https://github.com/conda-forge/miniforge for installation instructions.

pyqstrat relies on numpy, scipy and pandas which in turn use Fortran and C code that needs to be compiled. pyqstrat also includes C++ code that will need to be compiled

::

   mamba install pyqstrat

Requirements:

* Python_ version 3.10 or higher;

Documentation
-------------

The best way to get started is to go through the getting started Jupyter notebook: `Getting Started <https://github.com/abbass2/pyqstrat/tree/master/pyqstrat/notebooks/getting_started.ipynb>`_

`Jupyter Notebooks <https://github.com/abbass2/pyqstrat/tree/master/pyqstrat/notebooks>`_ 

`API docs <https://abbass2.github.io/pyqstrat>`_

Discussion
----------

The `pyqstrat user group <https://groups.io/g/pyqstrat>`_ is the group used for pyqstrat discussions. You can also add code issues via github


Disclaimer
----------

The software is provided on the conditions of the simplified BSD license.

.. _Python: http://www.python.org

.. |PyVersion| image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :alt:

.. |Status| image:: https://img.shields.io/badge/status-beta-green.svg
   :alt:

.. |License| image:: https://img.shields.io/badge/license-BSD-blue.svg
   :alt:
   
