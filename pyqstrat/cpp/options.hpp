#ifndef options_hpp
#define options_hpp

// Black Scholes options calculations from  www.jaeckel.org/LetsBeRational.7z

double pdf(double x);
double cdf(double x);
double d1(double S, double K, double t, double r, double sigma, double q = 0);
double d2(double S, double K, double t, double r, double sigma, double q = 0);
double black_scholes_price(bool call, double S, double K, double t, double r, double sigma, double q = 0);
double delta(bool call, double S, double K, double t, double r, double sigma, double q = 0);
double theta(bool call, double S, double K, double t, double r, double sigma, double q = 0);
// Use _ to avoid conflict with gamma in math.h
double _gamma(double S, double K, double t, double r, double sigma, double q = 0);
double vega(double S, double K, double t, double r, double sigma, double q = 0);
double rho(bool call, double S, double K, double t, double r, double sigma, double q = 0);
double implied_vol(bool call, double price, double S, double K, double t, double r, double q = 0);

// From lets be rational

#include "lets_be_rational/importexport.h"

EXPORT_EXTERN_C double implied_volatility_from_a_transformed_rational_guess(double price, double F, double K, double T, double q);

#endif

