#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>

#include "options.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

void init_pybind_options(py::module &m) {
     m.def("_pdf", py::vectorize(pdf), "x"_a,
           R"pqdoc(
           Probability density function of normal distribution
           Args:
               x (float): random variable
           Returns:
               float: pdf of the random variable
           )pqdoc");
    
    m.def("cdf", py::vectorize(cdf), "x"_a,
          R"pqdoc(
          Cumulative density function of normal distribution
          Args:
              x (float): random variable
          Returns:
              float: cdf of the random variable
          )pqdoc");
    
    m.def("d1", py::vectorize(d1),
          "S"_a,
          "K"_a,
          "t"_a,
          "r"_a,
          "sigma"_a,
          "q"_a = 0.0,
          R"pqdoc(
          d1 from Black Scholes
          Args:
              S (float): Spot price. For a future discount the future price using exp(-rt)
              K (float): Strike
              t (float): Time to maturity in years
              r (float): Continuously compounded interest rate.  Use 0.01 for 1%
              sigma (float): Annualized volatility.  Use 0.01 for 1%
              q (float): Annualized dividend yield.  Use 0.01 for 1%
          Returns:
            float:
          )pqdoc");
    
    m.def("d2", py::vectorize(d2),
          "S"_a,
          "K"_a,
          "t"_a,
          "r"_a,
          "sigma"_a,
          "q"_a = 0.0,
          R"pqdoc(
          d2 from Black Scholes
          Args:
              S (float): Spot price. For a future discount the future price using exp(-rt)
              K (float): Strike
              t (float): Time to maturity in years
              r (float): Continuously compounded interest rate.  Use 0.01 for 1%
              sigma (float): Annualized volatility.  Use 0.01 for 1%
              q (float): Annualized dividend yield.  Use 0.01 for 1%
          Returns:
            float:
          )pqdoc");
    
    m.def("black_scholes_price", py::vectorize(black_scholes_price),
          "call"_a,
          "S"_a,
          "K"_a,
          "t"_a,
          "r"_a,
          "sigma"_a,
          "q"_a = 0.0,
          R"pqdoc(
          Compute Euroepean option price
          Args:
              call (bool): True for a call option, False for a put
              S (float): Spot price. For a future discount the future price using exp(-rt)
              K (float): Strike
              t (float): Time to maturity in years
              r (float): Continuously compounded interest rate.  Use 0.01 for 1%
              sigma (float): Annualized volatility.  Use 0.01 for 1%
              q (float, optional): Annualized dividend yield.  Use 0.01 for 1%.  Default 0
          Returns:
              float: Option price
          )pqdoc");
    
    m.def("delta", py::vectorize(delta),
          "call"_a,
          "S"_a,
          "K"_a,
          "t"_a,
          "r"_a,
          "sigma"_a,
          "q"_a = 0.0,
          R"pqdoc(
          Compute European option delta
          Args:
              call (bool): True for a call option, False for a put
              S (float): Spot price. For a future discount the future price using exp(-rt)
              K (float): Strike
              t (float): Time to maturity in years
              r (float): Continuously compounded interest rate.  Use 0.01 for 1%
              sigma (float): Annualized volatility.  Use 0.01 for 1%
              q (float, optional): Annualized dividend yield.  Use 0.01 for 1%.  Default 0
          Returns:
              float: Option delta
          )pqdoc");
    
    m.def("theta", py::vectorize(theta),
          "call"_a,
          "F"_a,
          "K"_a,
          "t"_a,
          "r"_a,
          "sigma"_a,
          "q"_a = 0.0,
          R"pqdoc(
          Compute European option theta per day.  This is Black Scholes formula theta divided by 365 to give us the customary theta per day
          Args:
              call (bool): True for a call option, False for a put
              S (float): Spot price. For a future discount the future price using exp(-rt)
              K (float): Strike
              t (float): Time to maturity in years
              r (float): Continuously compounded interest rate.  Use 0.01 for 1%
              sigma (float): Annualized volatility.  Use 0.01 for 1%
              q (float, optional): Annualized dividend yield.  Use 0.01 for 1%.  Default 0
          Returns:
              float: Option theta
          )pqdoc");
    
    m.def("gamma", py::vectorize(_gamma),
          "S"_a,
          "K"_a,
          "t"_a,
          "r"_a,
          "sigma"_a,
          "q"_a = 0.0,
          R"pqdoc(
          Compute European option gamma.
          Args:
              S (float): Spot price. For a future discount the future price using exp(-rt)
              K (float): Strike
              t (float): Time to maturity in years
              r (float): Continuously compounded interest rate.  Use 0.01 for 1%
              sigma (float): Annualized volatility.  Use 0.01 for 1%
              q (float, optional): Annualized dividend yield.  Use 0.01 for 1%.  Default 0
          Returns:
              float: Option gamma
          )pqdoc");
    
    m.def("vega", &vega,
          "S"_a,
          "K"_a,
          "t"_a,
          "r"_a,
          "sigma"_a,
          "q"_a = 0.0,
          R"pqdoc(
          Compute European option vega.  This is Black Scholes formula vega divided by 100 so we get rho per 1% change in interest rate
          Args:
              S (float): Spot price. For a future discount the future price using exp(-rt)
              K (float): Strike
              t (float): Time to maturity in years
              r (float): Continuously compounded interest rate.  Use 0.01 for 1%
              sigma (float): Annualized volatility.  Use 0.01 for 1%
              q (float, optional): Annualized dividend yield.  Use 0.01 for 1%.  Default 0
          Returns:
              float: Option vega
          )pqdoc");
    
    m.def("rho", py::vectorize(theta),
          "call"_a,
          "S"_a,
          "K"_a,
          "t"_a,
          "r"_a,
          "sigma"_a,
          "q"_a = 0.0,
          R"pqdoc(
          Compute European option rho.  This is Black Scholes formula rho divided by 100 so we get rho per 1% change in interest rate
          Args:
              call (bool): True for a European call option, False for a put
              S (float): Spot price. For a future discount the future price using exp(-rt)
              K (float): Strike
              t (float): Time to maturity in years
              r (float): Continuously compounded interest rate.  Use 0.01 for 1%
              sigma (float): Annualized volatility.  Use 0.01 for 1%
              q (float, optional): Annualized dividend yield.  Use 0.01 for 1%.  Default 0
          Returns:
              float: Option theta
          )pqdoc");
    
    m.def("implied_vol", py::vectorize(implied_vol),
          "call"_a,
          "price"_a,
          "S"_a,
          "K"_a,
          "t"_a,
          "r"_a,
          "q"_a = 0.0,
          R"pqdoc(
          Compute implied volatility for a European option.
          Args:
              call (bool): True for a call option, False for a put
              price (float): The option premium
              S (float): Spot price. For a future discount the future price using exp(-rt)
              K (float): Strike
              t (float): Time to maturity in years
              r (float): Continuously compounded interest rate.  Use 0.01 for 1%
              q (float, optional): Annualized dividend yield.  Use 0.01 for 1%.  Default 0
          Returns:
              float: Implied volatility.  For 1% we return 0.01
          )pqdoc");
}



