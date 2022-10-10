//Adaoted from py_vollib black_scholes_merton analytical greeks
#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include "options.hpp"

using namespace std;

const double PI  = 3.14159265358979323846264338327950288; // From M_PI definition

// pdf
double pdf(double x) {
    return ( 1.0 / (sqrt( 2 * PI))) * exp(-0.5 * x * x);
}

// cdf
double cdf(double x) {
    return std::erfc(-x / std::sqrt(2)) /2;
}

double d1(double S, double K, double t, double r, double sigma, double q) {
    return (log(S / K) + ((r - q)+ 0.5 * sigma * sigma) * t) / (sigma * sqrt(t));
}

double d2(double S, double K, double t, double r, double sigma, double q) {
    return d1(S, K,  t, r, sigma, q) - sigma * sqrt(t);
}

double black_scholes_price(bool call, double S, double K, double t, double r, double sigma, double q) {
    if (call) {
        return cdf(d1(S, K, t, r, sigma, q)) * S * exp(-q * t) - cdf(d2(S, K, t, r, sigma, q)) * K * exp(-r * t);
    } else {
        return cdf(-d2(S, K, t, r, sigma, q)) * K * exp(-r * t) - cdf(-d1(S, K, t, r, sigma, q)) * S * exp(-q * t);
    }
}

double delta(bool call, double S, double K, double t, double r, double sigma, double q) {
    double D1 = d1(S, K, t, r, sigma, q);
    if (call)
        return exp(-q * t) * cdf(D1);
    else
        return -exp(-q * t) * cdf(-D1);
}

double theta(bool call, double S, double K, double t, double r, double sigma, double q) {
    //     The text book analytical formula does not divide by 365,
    // but in practice theta is defined as the change in price
    // for each day change in t, hence we divide by 365.
    
    double D1 = d1(S, K, t, r, sigma, q);
    double D2 = d2(S, K, t, r, sigma, q);
    double first_term = (S* exp(-q * t) * pdf(D1) * sigma) / (2 * sqrt(t));
    
    if (call) {
        double second_term = -q * S * exp(-q * t) * cdf(D1);
        double third_term = r * K * exp(-r * t) * cdf(D2);
        return - (first_term + second_term + third_term) / 365.0;
    } else {
        double second_term = -q * S * exp(-q * t) * cdf(-D1);
        double third_term = r * K * exp(-r * t) * cdf(-D2);
        return (-first_term + second_term + third_term) / 365.0;
    }
}

double _gamma(double S, double K, double t, double r, double sigma, double q) {
    double D1 = d1(S, K, t, r, sigma, q);
    double numerator = exp(-q * t) * pdf(D1);
    double denominator = S * sigma * sqrt(t);
    return numerator / denominator;
}

double vega(double S, double K, double t, double r, double sigma, double q) {
    //The text book analytical formula does not multiply by .01,
    //but in practice vega is defined as the change in price
    //for each 1 percent change in IV, hence we multiply by 0.01.
    double D1 = d1(S, K, t, r, sigma, q);
    return S * exp(-q * t) * pdf(D1) * sqrt(t) * 0.01;
}

double rho(bool call, double S, double K, double t, double r, double sigma, double q) {
    //The text book analytical formula does not multiply by .01,
    //but in practice rho is defined as the change in price
    //for each 1 percent change in r, hence we multiply by 0.01.
    double D2 = d2(S, K, t, r, sigma, q);
    if (call)
        return t * K * exp(-r * t) * cdf(D2) * .01;
    else
        return -t * K * exp(-r * t) * cdf(-D2) * .01;
}

double implied_vol(bool call, double price, double S, double K, double t, double r, double q) {
    double dcf = exp(-1.0 * r * t);
    double undiscounted_price = price / dcf;
    double F = S * exp((r - q) * t); // Compute forward price
    double iv = implied_volatility_from_a_transformed_rational_guess(undiscounted_price, F, K, t, call ? 1.0 : -1.0);
    if (iv == std::numeric_limits<double>::max() || iv == -std::numeric_limits<double>::max()) return NAN;
    
    //cout << "iv: " << iv << " undiscounted premium: " << undiscounted_premium << " forward_or_spot: " << forward_or_spot <<
    //" strike: " << strike << " t: " << t << " put_call_flag: " << put_call_flag << endl;
    return iv;
}

int test_options() {
    double S = 9.3;
    double K = 10.0;
    double r = 0.01;
    double t = 1.0;
    double q = 0.01;
    double p = 1.0;
    bool call = false;
    double sigma = implied_vol(call, p, S, K, t, r, q);
    std::cout << std::fixed;
    std::cout << std::setprecision(10);
    
    cout << "implied vol: " << sigma << endl;
    sigma = 0.15;
    cout << "price: " << black_scholes_price(call, S, K, t, r, sigma, q) << endl;
    cout << "delta: " << delta(call, S, K, t, r, sigma, q) << endl;
    cout << "gamma: " << _gamma(S, K, t, r, sigma, q) << endl;
    cout << "theta: " << theta(call, S, K, t, r, sigma, q) << endl;
    cout << "vega: " << vega(S, K, t, r, sigma, q) << endl;
    cout << "rho: " << rho(call, S, K, t, r, sigma, q) << endl;
    return 0;
}

