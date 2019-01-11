//
// This source code resides at www.jaeckel.org/LetsBeRational.7z .
//
// ======================================================================================
// Copyright © 2013-2014 Peter Jäckel.
// 
// Permission to use, copy, modify, and distribute this software is freely granted,
// provided that this notice is preserved.
//
// WARRANTY DISCLAIMER
// The Software is provided "as is" without warranty of any kind, either express or implied,
// including without limitation any implied warranties of condition, uninterrupted use,
// merchantability, fitness for a particular purpose, or non-infringement.
// ======================================================================================
//
#ifndef   NORMAL_DISTRIBUTION_H
#define   NORMAL_DISTRIBUTION_H

#include <math.h>
#include <cmath>
#include "importexport.h"

#define ONE_OVER_SQRT_TWO     0.7071067811865475244008443621048490392848359376887
#define ONE_OVER_SQRT_TWO_PI  0.3989422804014326779399460599343818684758586311649
#define SQRT_TWO_PI           2.506628274631000502415765284811045253006986740610

EXPORT_EXTERN_C double erf_cody(double z);
EXPORT_EXTERN_C double erfc_cody(double z);
EXPORT_EXTERN_C double erfcx_cody(double z);
EXPORT_EXTERN_C double norm_cdf(double z);
inline double norm_pdf(double x){ return ONE_OVER_SQRT_TWO_PI*exp(-.5*x*x); }
EXPORT_EXTERN_C double inverse_norm_cdf(double u);

#endif // NORMAL_DISTRIBUTION_H
