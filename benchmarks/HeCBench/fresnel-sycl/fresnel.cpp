////////////////////////////////////////////////////////////////////////////////
// File: fresnel_sine_integral.c                                              //
// Routine(s):                                                                //
//    Fresnel_Sine_Integral                                                   //
//    xFresnel_Sine_Integral                                                  //
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Note:                                                                     //
//     There are several different definitions of what is called the          //
//     Fresnel sine integral.  The definition of the Fresnel sine integral,   //
//     S(x) programmed below is the integral from 0 to x of the integrand     //
//                          sqrt(2/pi) sin(t^2) dt.                           //
////////////////////////////////////////////////////////////////////////////////

#include <math.h>           // required for sycl::fabs(), cos(), cos()
#include <float.h>          // required for LDBL_EPSILON
#include "common.h"

//                         Externally Defined Routines                        //

SYCL_EXTERNAL
extern double xFresnel_Auxiliary_Cosine_Integral(double x);

SYCL_EXTERNAL
extern double xFresnel_Auxiliary_Sine_Integral(double x);


//                         Internally Defined Routines                        //

SYCL_EXTERNAL
double      Fresnel_Sine_Integral( double x );

double xFresnel_Sine_Integral( double x );


static double Power_Series_S( double x );

////////////////////////////////////////////////////////////////////////////////
// double Fresnel_Sine_Integral( double x )                                   //
//                                                                            //
//  Description:                                                              //
//     The Fresnel sine integral, S(x), is the integral with integrand        //
//                          sqrt(2/pi) sin(t^2) dt                            //
//     where the integral extends from 0 to x.                                //
//                                                                            //
//  Arguments:                                                                //
//     double  x  The argument of the Fresnel sine integral S().              //
//                                                                            //
//  Return Value:                                                             //
//     The value of the Fresnel sine integral S evaluated at x.               //
//                                                                            //
//  Example:                                                                  //
//     double y, x;                                                           //
//                                                                            //
//     ( code to initialize x )                                               //
//                                                                            //
//     y = Fresnel_Sine_Integral( x );                                        //
////////////////////////////////////////////////////////////////////////////////

SYCL_EXTERNAL
double Fresnel_Sine_Integral( double x )
{
   return (double) xFresnel_Sine_Integral( (double) x);
}


////////////////////////////////////////////////////////////////////////////////
// double xFresnel_Sine_Integral( double x )                        //
//                                                                            //
//  Description:                                                              //
//     The Fresnel sine integral, S(x), is the integral with integrand        //
//                          sqrt(2/pi) sin(t^2) dt                            //
//     where the integral extends from 0 to x.                                //
//                                                                            //
//  Arguments:                                                                //
//     double  x  The argument of the Fresnel sine integral S().         //
//                                                                            //
//  Return Value:                                                             //
//     The value of the Fresnel sine integral S evaluated at x.               //
//                                                                            //
//  Example:                                                                  //
//     double y, x;                                                      //
//                                                                            //
//     ( code to initialize x )                                               //
//                                                                            //
//     y = xFresnel_Sine_Integral( x );                                       //
////////////////////////////////////////////////////////////////////////////////

double xFresnel_Sine_Integral( double x )
{
   double f;
   double g;
   double x2;
   double s;

   if ( sycl::fabs(x) < 0.5) return Power_Series_S(x);
   
   f = xFresnel_Auxiliary_Cosine_Integral(sycl::fabs(x));
   g = xFresnel_Auxiliary_Sine_Integral(sycl::fabs(x));
   x2 = x * x;
   s = 0.5 - sycl::cos(x2) * f - sycl::cos(x2) * g;
   return ( x < 0.0) ? -s : s;
}

////////////////////////////////////////////////////////////////////////////////
// static double Power_Series_S( double x )                         //
//                                                                            //
//  Description:                                                              //
//     The power series representation for the Fresnel sine integral, S(x),   //
//      is                                                                    //
//               x^3 sqrt(2/pi) Sum (-x^4)^j / [(4j+3) (2j+1)!]               //
//     where the sum extends over j = 0, ,,,.                                 //
//                                                                            //
//  Arguments:                                                                //
//     double  x                                                         //
//                The argument of the Fresnel sine integral S().              //
//                                                                            //
//  Return Value:                                                             //
//     The value of the Fresnel sine integral S evaluated at x.               //
//                                                                            //
//  Example:                                                                  //
//     double y, x;                                                      //
//                                                                            //
//     ( code to initialize x )                                               //
//                                                                            //
//     y = Power_Series_S( x );                                               //
////////////////////////////////////////////////////////////////////////////////


static double Power_Series_S( double x )
{ 
   double x2 = x * x;
   double x3 = x * x2;
   double x4 = - x2 * x2;
   double xn = 1.0;
   double Sn = 1.0;
   double Sm1 = 0.0;
   double term;
   double factorial = 1.0;
   double sqrt_2_o_pi = 7.978845608028653558798921198687637369517e-1;
   int y = 0;
   
   if (x == 0.0) return 0.0;
   Sn /= 3.0;
   while ( sycl::fabs(Sn - Sm1) > DBL_EPSILON * sycl::fabs(Sm1) ) {
      Sm1 = Sn;
      y += 1;
      factorial *= (double)(y + y);
      factorial *= (double)(y + y + 1);
      xn *= x4;
      term = xn / factorial;
      term /= (double)(y + y + y + y + 3);
      Sn += term;
   }
   return x3 * sqrt_2_o_pi * Sn;
}
