//
// Original Fortran code taken from http://www.netlib.org/specfun/erf, compiled with f2c, and adapted by hand.
//
// Created with command line f2c -C++ -c -a -krd -r8 cody_erf.f
//
// Translated by f2c (version 20100827).
//

//
// This source code resides at www.jaeckel.org/LetsBeRational.7z .
//
// ======================================================================================
// WARRANTY DISCLAIMER
// The Software is provided "as is" without warranty of any kind, either express or implied,
// including without limitation any implied warranties of condition, uninterrupted use,
// merchantability, fitness for a particular purpose, or non-infringement.
// ======================================================================================
//

#if defined( _DEBUG ) || defined( BOUNDS_CHECK_STL_ARRAYS )
#define _SECURE_SCL 1
#define _SECURE_SCL_THROWS 1
#define _SCL_SECURE_NO_WARNINGS
#define _HAS_ITERATOR_DEBUGGING 0
#else
#define _SECURE_SCL 0
#endif
#if defined(_MSC_VER)
# define NOMINMAX // to suppress MSVC's definitions of min() and max()
// These four pragmas are the equivalent to /fp:fast.
# pragma float_control( except, off )
# pragma float_control( precise, off )
# pragma fp_contract( on )
# pragma fenv_access( off )
#endif

#include "normaldistribution.h"
#include <math.h>
#include <float.h>

namespace {
   inline double d_int(const double x){ return( (x>0) ? floor(x) : -floor(-x) ); }
}

/*<       SUBROUTINE CALERF(ARG,RESULT,JINT) >*/
double calerf(double x, const int jint) {

   static const double a[5] = { 3.1611237438705656,113.864154151050156,377.485237685302021,3209.37758913846947,.185777706184603153 };
   static const double b[4] = { 23.6012909523441209,244.024637934444173,1282.61652607737228,2844.23683343917062 };
   static const double c__[9] = { .564188496988670089,8.88314979438837594,66.1191906371416295,298.635138197400131,881.95222124176909,1712.04761263407058,2051.07837782607147,1230.33935479799725,2.15311535474403846e-8 };
   static const double d__[8] = { 15.7449261107098347,117.693950891312499,537.181101862009858,1621.38957456669019,3290.79923573345963,4362.61909014324716,3439.36767414372164,1230.33935480374942 };
   static const double p[6] = { .305326634961232344,.360344899949804439,.125781726111229246,.0160837851487422766,6.58749161529837803e-4,.0163153871373020978 };
   static const double q[5] = { 2.56852019228982242,1.87295284992346047,.527905102951428412,.0605183413124413191,.00233520497626869185 };

   static const double zero = 0.;
   static const double half = .5;
   static const double one = 1.;
   static const double two = 2.;
   static const double four = 4.;
   static const double sqrpi = 0.56418958354775628695;
   static const double thresh = .46875;
   static const double sixten = 16.;

   double y, del, ysq, xden, xnum, result;

   /* ------------------------------------------------------------------ */
   /* This packet evaluates  erf(x),  erfc(x),  and  exp(x*x)*erfc(x) */
   /*   for a real argument  x.  It contains three FUNCTION type */
   /*   subprograms: ERF, ERFC, and ERFCX (or DERF, DERFC, and DERFCX), */
   /*   and one SUBROUTINE type subprogram, CALERF.  The calling */
   /*   statements for the primary entries are: */
   /*                   Y=ERF(X)     (or   Y=DERF(X)), */
   /*                   Y=ERFC(X)    (or   Y=DERFC(X)), */
   /*   and */
   /*                   Y=ERFCX(X)   (or   Y=DERFCX(X)). */
   /*   The routine  CALERF  is intended for internal packet use only, */
   /*   all computations within the packet being concentrated in this */
   /*   routine.  The function subprograms invoke  CALERF  with the */
   /*   statement */
   /*          CALL CALERF(ARG,RESULT,JINT) */
   /*   where the parameter usage is as follows */
   /*      Function                     Parameters for CALERF */
   /*       call              ARG                  Result          JINT */
   /*     ERF(ARG)      ANY REAL ARGUMENT         ERF(ARG)          0 */
   /*     ERFC(ARG)     ABS(ARG) .LT. XBIG        ERFC(ARG)         1 */
   /*     ERFCX(ARG)    XNEG .LT. ARG .LT. XMAX   ERFCX(ARG)        2 */
   /*   The main computation evaluates near-minimax approximations */
   /*   from "Rational Chebyshev approximations for the error function" */
   /*   by W. J. Cody, Math. Comp., 1969, PP. 631-638.  This */
   /*   transportable program uses rational functions that theoretically */
   /*   approximate  erf(x)  and  erfc(x)  to at least 18 significant */
   /*   decimal digits.  The accuracy achieved depends on the arithmetic */
   /*   system, the compiler, the intrinsic functions, and proper */
   /*   selection of the machine-dependent constants. */
   /* ******************************************************************* */
   /* ******************************************************************* */
   /* Explanation of machine-dependent constants */
   /*   XMIN   = the smallest positive floating-point number. */
   /*   XINF   = the largest positive finite floating-point number. */
   /*   XNEG   = the largest negative argument acceptable to ERFCX; */
   /*            the negative of the solution to the equation */
   /*            2*exp(x*x) = XINF. */
   /*   XSMALL = argument below which erf(x) may be represented by */
   /*            2*x/sqrt(pi)  and above which  x*x  will not underflow. */
   /*            A conservative value is the largest machine number X */
   /*            such that   1.0 + X = 1.0   to machine precision. */
   /*   XBIG   = largest argument acceptable to ERFC;  solution to */
   /*            the equation:  W(x) * (1-0.5/x**2) = XMIN,  where */
   /*            W(x) = exp(-x*x)/[x*sqrt(pi)]. */
   /*   XHUGE  = argument above which  1.0 - 1/(2*x*x) = 1.0  to */
   /*            machine precision.  A conservative value is */
   /*            1/[2*sqrt(XSMALL)] */
   /*   XMAX   = largest acceptable argument to ERFCX; the minimum */
   /*            of XINF and 1/[sqrt(pi)*XMIN]. */
   // The numbers below were preselected for IEEE .
   static const double xinf = 1.79e308;
   static const double xneg = -26.628;
   static const double xsmall = 1.11e-16;
   static const double xbig = 26.543;
   static const double xhuge = 6.71e7;
   static const double xmax = 2.53e307;
   /*   Approximate values for some important machines are: */
   /*                          XMIN       XINF        XNEG     XSMALL */
   /*  CDC 7600      (S.P.)  3.13E-294   1.26E+322   -27.220  7.11E-15 */
   /*  CRAY-1        (S.P.)  4.58E-2467  5.45E+2465  -75.345  7.11E-15 */
   /*  IEEE (IBM/XT, */
   /*    SUN, etc.)  (S.P.)  1.18E-38    3.40E+38     -9.382  5.96E-8 */
   /*  IEEE (IBM/XT, */
   /*    SUN, etc.)  (D.P.)  2.23D-308   1.79D+308   -26.628  1.11D-16 */
   /*  IBM 195       (D.P.)  5.40D-79    7.23E+75    -13.190  1.39D-17 */
   /*  UNIVAC 1108   (D.P.)  2.78D-309   8.98D+307   -26.615  1.73D-18 */
   /*  VAX D-Format  (D.P.)  2.94D-39    1.70D+38     -9.345  1.39D-17 */
   /*  VAX G-Format  (D.P.)  5.56D-309   8.98D+307   -26.615  1.11D-16 */
   /*                          XBIG       XHUGE       XMAX */
   /*  CDC 7600      (S.P.)  25.922      8.39E+6     1.80X+293 */
   /*  CRAY-1        (S.P.)  75.326      8.39E+6     5.45E+2465 */
   /*  IEEE (IBM/XT, */
   /*    SUN, etc.)  (S.P.)   9.194      2.90E+3     4.79E+37 */
   /*  IEEE (IBM/XT, */
   /*    SUN, etc.)  (D.P.)  26.543      6.71D+7     2.53D+307 */
   /*  IBM 195       (D.P.)  13.306      1.90D+8     7.23E+75 */
   /*  UNIVAC 1108   (D.P.)  26.582      5.37D+8     8.98D+307 */
   /*  VAX D-Format  (D.P.)   9.269      1.90D+8     1.70D+38 */
   /*  VAX G-Format  (D.P.)  26.569      6.71D+7     8.98D+307 */
   /* ******************************************************************* */
   /* ******************************************************************* */
   /* Error returns */
   /*  The program returns  ERFC = 0      for  ARG .GE. XBIG; */
   /*                       ERFCX = XINF  for  ARG .LT. XNEG; */
   /*      and */
   /*                       ERFCX = 0     for  ARG .GE. XMAX. */
   /* Intrinsic functions required are: */
   /*     ABS, AINT, EXP */
   /*  Author: W. J. Cody */
   /*          Mathematics and Computer Science Division */
   /*          Argonne National Laboratory */
   /*          Argonne, IL 60439 */
   /*  Latest modification: March 19, 1990 */
   /* ------------------------------------------------------------------ */
   /*<       INTEGER I,JINT >*/
   /* S    REAL */
   /*<    >*/
   /*<       DIMENSION A(5),B(4),C(9),D(8),P(6),Q(5) >*/
   /* ------------------------------------------------------------------ */
   /*  Mathematical constants */
   /* ------------------------------------------------------------------ */
   /* S    DATA FOUR,ONE,HALF,TWO,ZERO/4.0E0,1.0E0,0.5E0,2.0E0,0.0E0/, */
   /* S   1     SQRPI/5.6418958354775628695E-1/,THRESH/0.46875E0/, */
   /* S   2     SIXTEN/16.0E0/ */
   /*<    >*/
   /* ------------------------------------------------------------------ */
   /*  Machine-dependent constants */
   /* ------------------------------------------------------------------ */
   /* S    DATA XINF,XNEG,XSMALL/3.40E+38,-9.382E0,5.96E-8/, */
   /* S   1     XBIG,XHUGE,XMAX/9.194E0,2.90E3,4.79E37/ */
   /*<    >*/
   /* ------------------------------------------------------------------ */
   /*  Coefficients for approximation to  erf  in first interval */
   /* ------------------------------------------------------------------ */
   /* S    DATA A/3.16112374387056560E00,1.13864154151050156E02, */
   /* S   1       3.77485237685302021E02,3.20937758913846947E03, */
   /* S   2       1.85777706184603153E-1/ */
   /* S    DATA B/2.36012909523441209E01,2.44024637934444173E02, */
   /* S   1       1.28261652607737228E03,2.84423683343917062E03/ */
   /*<    >*/
   /*<    >*/
   /* ------------------------------------------------------------------ */
   /*  Coefficients for approximation to  erfc  in second interval */
   /* ------------------------------------------------------------------ */
   /* S    DATA C/5.64188496988670089E-1,8.88314979438837594E0, */
   /* S   1       6.61191906371416295E01,2.98635138197400131E02, */
   /* S   2       8.81952221241769090E02,1.71204761263407058E03, */
   /* S   3       2.05107837782607147E03,1.23033935479799725E03, */
   /* S   4       2.15311535474403846E-8/ */
   /* S    DATA D/1.57449261107098347E01,1.17693950891312499E02, */
   /* S   1       5.37181101862009858E02,1.62138957456669019E03, */
   /* S   2       3.29079923573345963E03,4.36261909014324716E03, */
   /* S   3       3.43936767414372164E03,1.23033935480374942E03/ */
   /*<    >*/
   /*<    >*/
   /* ------------------------------------------------------------------ */
   /*  Coefficients for approximation to  erfc  in third interval */
   /* ------------------------------------------------------------------ */
   /* S    DATA P/3.05326634961232344E-1,3.60344899949804439E-1, */
   /* S   1       1.25781726111229246E-1,1.60837851487422766E-2, */
   /* S   2       6.58749161529837803E-4,1.63153871373020978E-2/ */
   /* S    DATA Q/2.56852019228982242E00,1.87295284992346047E00, */
   /* S   1       5.27905102951428412E-1,6.05183413124413191E-2, */
   /* S   2       2.33520497626869185E-3/ */
   /*<    >*/
   /*<    >*/
   /* ------------------------------------------------------------------ */
   /*<       X = ARG >*/
   // x = *arg;
   /*<       Y = ABS(X) >*/
   y = fabs(x);
   /*<       IF (Y .LE. THRESH) THEN >*/
   if (y <= thresh) {
      /* ------------------------------------------------------------------ */
      /*  Evaluate  erf  for  |X| <= 0.46875 */
      /* ------------------------------------------------------------------ */
      /*<             YSQ = ZERO >*/
      ysq = zero;
      /*<             IF (Y .GT. XSMALL) YSQ = Y * Y >*/
      if (y > xsmall) {
         ysq = y * y;
      }
      /*<             XNUM = A(5)*YSQ >*/
      xnum = a[4] * ysq;
      /*<             XDEN = YSQ >*/
      xden = ysq;
      /*<             DO 20 I = 1, 3 >*/
      for (int i__ = 1; i__ <= 3; ++i__) {
         /*<                XNUM = (XNUM + A(I)) * YSQ >*/
         xnum = (xnum + a[i__ - 1]) * ysq;
         /*<                XDEN = (XDEN + B(I)) * YSQ >*/
         xden = (xden + b[i__ - 1]) * ysq;
         /*<    20       CONTINUE >*/
         /* L20: */
      }
      /*<             RESULT = X * (XNUM + A(4)) / (XDEN + B(4)) >*/
      result = x * (xnum + a[3]) / (xden + b[3]);
      /*<             IF (JINT .NE. 0) RESULT = ONE - RESULT >*/
      if (jint != 0) {
         result = one - result;
      }
      /*<             IF (JINT .EQ. 2) RESULT = EXP(YSQ) * RESULT >*/
      if (jint == 2) {
         result = exp(ysq) * result;
      }
      /*<             GO TO 800 >*/
      goto L800;
      /* ------------------------------------------------------------------ */
      /*  Evaluate  erfc  for 0.46875 <= |X| <= 4.0 */
      /* ------------------------------------------------------------------ */
      /*<          ELSE IF (Y .LE. FOUR) THEN >*/
   } else if (y <= four) {
      /*<             XNUM = C(9)*Y >*/
      xnum = c__[8] * y;
      /*<             XDEN = Y >*/
      xden = y;
      /*<             DO 120 I = 1, 7 >*/
      for (int i__ = 1; i__ <= 7; ++i__) {
         /*<                XNUM = (XNUM + C(I)) * Y >*/
         xnum = (xnum + c__[i__ - 1]) * y;
         /*<                XDEN = (XDEN + D(I)) * Y >*/
         xden = (xden + d__[i__ - 1]) * y;
         /*<   120       CONTINUE >*/
         /* L120: */
      }
      /*<             RESULT = (XNUM + C(8)) / (XDEN + D(8)) >*/
      result = (xnum + c__[7]) / (xden + d__[7]);
      /*<             IF (JINT .NE. 2) THEN >*/
      if (jint != 2) {
         /*<                YSQ = AINT(Y*SIXTEN)/SIXTEN >*/
         double d__1 = y * sixten;
         ysq = d_int(d__1) / sixten;
         /*<                DEL = (Y-YSQ)*(Y+YSQ) >*/
         del = (y - ysq) * (y + ysq);
         /*<                RESULT = EXP(-YSQ*YSQ) * EXP(-DEL) * RESULT >*/
         d__1 = exp(-ysq * ysq) * exp(-del);
         result = d__1 * result;
         /*<             END IF >*/
      }
      /* ------------------------------------------------------------------ */
      /*  Evaluate  erfc  for |X| > 4.0 */
      /* ------------------------------------------------------------------ */
      /*<          ELSE >*/
   } else {
      /*<             RESULT = ZERO >*/
      result = zero;
      /*<             IF (Y .GE. XBIG) THEN >*/
      if (y >= xbig) {
         /*<                IF ((JINT .NE. 2) .OR. (Y .GE. XMAX)) GO TO 300 >*/
         if (jint != 2 || y >= xmax) {
            goto L300;
         }
         /*<                IF (Y .GE. XHUGE) THEN >*/
         if (y >= xhuge) {
            /*<                   RESULT = SQRPI / Y >*/
            result = sqrpi / y;
            /*<                   GO TO 300 >*/
            goto L300;
            /*<                END IF >*/
         }
         /*<             END IF >*/
      }
      /*<             YSQ = ONE / (Y * Y) >*/
      ysq = one / (y * y);
      /*<             XNUM = P(6)*YSQ >*/
      xnum = p[5] * ysq;
      /*<             XDEN = YSQ >*/
      xden = ysq;
      /*<             DO 240 I = 1, 4 >*/
      for (int i__ = 1; i__ <= 4; ++i__) {
         /*<                XNUM = (XNUM + P(I)) * YSQ >*/
         xnum = (xnum + p[i__ - 1]) * ysq;
         /*<                XDEN = (XDEN + Q(I)) * YSQ >*/
         xden = (xden + q[i__ - 1]) * ysq;
         /*<   240       CONTINUE >*/
         /* L240: */
      }
      /*<             RESULT = YSQ *(XNUM + P(5)) / (XDEN + Q(5)) >*/
      result = ysq * (xnum + p[4]) / (xden + q[4]);
      /*<             RESULT = (SQRPI -  RESULT) / Y >*/
      result = (sqrpi - result) / y;
       /**<             IF (JINT .NE. 2) THEN >*/
      if (jint != 2) {
         /*<                YSQ = AINT(Y*SIXTEN)/SIXTEN >*/
         double d__1 = y * sixten;
         ysq = d_int(d__1) / sixten;
         /*<                DEL = (Y-YSQ)*(Y+YSQ) >*/
         del = (y - ysq) * (y + ysq);
         /*<                RESULT = EXP(-YSQ*YSQ) * EXP(-DEL) * RESULT >*/
         d__1 = exp(-ysq * ysq) * exp(-del);
         result = d__1 * result;
         /*<             END IF >*/
      }
      /*<       END IF >*/
   }
   /* ------------------------------------------------------------------ */
   /*  Fix up for negative argument, erf, etc. */
   /* ------------------------------------------------------------------ */
   /*<   300 IF (JINT .EQ. 0) THEN >*/
L300:
   if (jint == 0) {
      /*<             RESULT = (HALF - RESULT) + HALF >*/
      result = (half - result) + half;
      /*<             IF (X .LT. ZERO) RESULT = -RESULT >*/
      if (x < zero) {
         result = -(result);
      }
      /*<          ELSE IF (JINT .EQ. 1) THEN >*/
   } else if (jint == 1) {
      /*<             IF (X .LT. ZERO) RESULT = TWO - RESULT >*/
      if (x < zero) {
         result = two - result;
      }
      /*<          ELSE >*/
   } else {
      /*<             IF (X .LT. ZERO) THEN >*/
      if (x < zero) {
         /*<                IF (X .LT. XNEG) THEN >*/
         if (x < xneg) {
            /*<                      RESULT = XINF >*/
            result = xinf;
            /*<                   ELSE >*/
         } else {
            /*<                      YSQ = AINT(X*SIXTEN)/SIXTEN >*/
            double d__1 = x * sixten;
            ysq = d_int(d__1) / sixten;
            /*<                      DEL = (X-YSQ)*(X+YSQ) >*/
            del = (x - ysq) * (x + ysq);
            /*<                      Y = EXP(YSQ*YSQ) * EXP(DEL) >*/
            y = exp(ysq * ysq) * exp(del);
            /*<                      RESULT = (Y+Y) - RESULT >*/
            result = y + y - result;
            /*<                END IF >*/
         }
         /*<             END IF >*/
      }
      /*<       END IF >*/
   }
   /*<   800 RETURN >*/
L800:
   return result;
   /* ---------- Last card of CALERF ---------- */
   /*<       END >*/
} /* calerf_ */

/* S    REAL FUNCTION ERF(X) */
/*<       DOUBLE PRECISION FUNCTION DERF(X) >*/
double erf_cody(double x){
   /* -------------------------------------------------------------------- */
   /* This subprogram computes approximate values for erf(x). */
   /*   (see comments heading CALERF). */
   /*   Author/date: W. J. Cody, January 8, 1985 */
   /* -------------------------------------------------------------------- */
   /*<       INTEGER JINT >*/
   /* S    REAL             X, RESULT */
   /*<       DOUBLE PRECISION X, RESULT >*/
   /* ------------------------------------------------------------------ */
   /*<       JINT = 0 >*/
   /*<       CALL CALERF(X,RESULT,JINT) >*/
   return calerf(x, 0);
   /* S    ERF = RESULT */
   /*<       DERF = RESULT >*/
   /*<       RETURN >*/
   /* ---------- Last card of DERF ---------- */
   /*<       END >*/
} /* derf_ */

/* S    REAL FUNCTION ERFC(X) */
/*<       DOUBLE PRECISION FUNCTION DERFC(X) >*/
double erfc_cody(double x) {
   /* -------------------------------------------------------------------- */
   /* This subprogram computes approximate values for erfc(x). */
   /*   (see comments heading CALERF). */
   /*   Author/date: W. J. Cody, January 8, 1985 */
   /* -------------------------------------------------------------------- */
   /*<       INTEGER JINT >*/
   /* S    REAL             X, RESULT */
   /*<       DOUBLE PRECISION X, RESULT >*/
   /* ------------------------------------------------------------------ */
   /*<       JINT = 1 >*/
   /*<       CALL CALERF(X,RESULT,JINT) >*/
   return calerf(x, 1);
   /* S    ERFC = RESULT */
   /*<       DERFC = RESULT >*/
   /*<       RETURN >*/
   /* ---------- Last card of DERFC ---------- */
   /*<       END >*/
} /* derfc_ */

/* S    REAL FUNCTION ERFCX(X) */
/*<       DOUBLE PRECISION FUNCTION DERFCX(X) >*/
double erfcx_cody(double x) {
   /* ------------------------------------------------------------------ */
   /* This subprogram computes approximate values for exp(x*x) * erfc(x). */
   /*   (see comments heading CALERF). */
   /*   Author/date: W. J. Cody, March 30, 1987 */
   /* ------------------------------------------------------------------ */
   /*<       INTEGER JINT >*/
   /* S    REAL             X, RESULT */
   /*<       DOUBLE PRECISION X, RESULT >*/
   /* ------------------------------------------------------------------ */
   /*<       JINT = 2 >*/
   /*<       CALL CALERF(X,RESULT,JINT) >*/
   return calerf(x, 2);
   /* S    ERFCX = RESULT */
   /*<       DERFCX = RESULT >*/
   /*<       RETURN >*/
   /* ---------- Last card of DERFCX ---------- */
   /*<       END >*/
} /* derfcx_ */
