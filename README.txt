|Group| |PyVersion| |Status| |PyPiVersion| |License|

Introduction
============

The ``pyqstrat`` package is designed for backtesting quantitative strategies.  It was originally built for my own use as a quant trader / researcher, after I could not find a python based framework that was fast, extensible and transparent enough for use in my work.  It's designed for capable programmers who are comfortable with numpy and reasonably advanced Python techniques.

The goals are:

* Speed - Performance sensitive components are written at the numpy level which can lead to performance improvement of several orders of magnitude over python code.  Where possible, we parrallelize work so you can take advantage of all the cores available on your machine.
* Transparency - If you are going to commit money to a strategy, you want to know exactly what assumptions you are making.  The code is written and documented so these are as clear as possible.
* Extensibility - It would be impossible to think of all requirements for backtesting strategies that traders could come up with.  In addition, traders will want to measure different depending on the strategy being traded.

The framework is designed 

Using this framework, you can:

* Construct a portfolio containing multiple strategies that run concurrently
* Construct arbitrary indicators, trading signals and trading rules and add them to a strategy
* Add multiple symbols representing real or "virtual" instruments to a strategy
* Reuse existing market simulation or build your own to simulate how and when orders are filled
* Measure well known return metrics such as sharpe, calmar and also add your own metrics.
* Resample market data bars into lower frequencies
* Plot trades, market data, indicators and add custom subplots to give you insight into your strategy's operation.
* Optimize your strategy's parameters using multiple processes running concurrently.

Installation
------------

::

    pip3 install -U pyqstrat

Requirements:

* Python_ version 3.6 or higher;


Documentation
-------------

`notebooks <http://rawgit.com/saabbasi/pyqstrat/master/docs/html/notebooks.html>`_
`API docs <http://rawgit.com/saabbasi/pyqstrat/master/docs/html/api.html>`_

Discussion
----------

The `pyqstrat user group <https://groups.io/g/pyqstrat>`_ is the group used for pyqstrat discussions.


Acknowledgements
----------------

Before building this, I looked at the following.  Although I ended up not using them, I got some great ideas for them.

`R quantstrat library <https://github.com/braverock/quantstrat>`_
`Python backtrader project <https://www.backtrader.com>`_


Here are some books I recommend for learning to build quant strategies.  Several of the ideas I use in this framework come from these books

`Trading Systems: A New Approach to System Development and Portfolio Optimisation - Tomasini, Emilio and Jaekle, Urban <https://www.amazon.com/gp/product/1905641796/ref=oh_aui_search_detailpage?ie=UTF8&psc=1>`_
`Machine Trading - Chan, Ernie <https://www.amazon.com/gp/product/1119219604>`_
`Algorithmic Trading: Winning Strategies and Their Rationale - Chan, Ernie https://www.amazon.com/gp/product/1118460146>_`

Disclaimer
----------

The software is provided on the conditions of the simplified BSD license.

.. _Python: http://www.python.org
.. _`Interactive Brokers Python API`: http://interactivebrokers.github.io

.. |Group| image:: https://img.shields.io/badge/groups.io-insync-green.svg
   :alt: Join the user group
   :target: https://groups.io/g/insync

.. |PyPiVersion| image:: https://img.shields.io/pypi/v/pyqstrat.svg
   :alt: PyPi
   :target: https://pypi.python.org/pypi/pyqstrat

.. |PyVersion| image:: https://img.shields.io/badge/python-3.6+-blue.svg
   :alt:

.. |Status| image:: https://img.shields.io/badge/status-beta-green.svg
   :alt:

.. |License| image:: https://img.shields.io/badge/license-BSD-blue.svg
   :alt:
   
.. |Docs| image:: https://readthedocs.org/projects/pyqstrat/badge/?version=latest
   :alt: Documentation Status
   :target: http://rawgit.com/saabbasi/pyqstrat/master/docs/html/api.html
