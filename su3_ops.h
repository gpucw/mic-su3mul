#ifndef _SU3_OPS_H
#define _SU3_OPS_H 1
#include <types.h>

__inline__ void 
u_eq_u(complex *w, complex *u)
{
  (w+0)->re = (u+0)->re;
  (w+0)->im = (u+0)->im;
  (w+1)->re = (u+1)->re;
  (w+1)->im = (u+1)->im;
  (w+2)->re = (u+2)->re;
  (w+2)->im = (u+2)->im;
  (w+3)->re = (u+3)->re;
  (w+3)->im = (u+3)->im;
  (w+4)->re = (u+4)->re;
  (w+4)->im = (u+4)->im;
  (w+5)->re = (u+5)->re;
  (w+5)->im = (u+5)->im;
  (w+6)->re = (u+6)->re;
  (w+6)->im = (u+6)->im;
  (w+7)->re = (u+7)->re;
  (w+7)->im = (u+7)->im;
  (w+8)->re = (u+8)->re;
  (w+8)->im = (u+8)->im;
  return;
}

__inline__ void 
u_eq_uxu(complex *w, complex *u, complex *v)
{
  (w+0)->re = (u+0)->re*(v+0)->re + (u+1)->re*(v+3)->re + (u+2)->re*(v+6)->re - (u+0)->im*(v+0)->im - (u+1)->im*(v+3)->im - (u+2)->im*(v+6)->im;
  (w+0)->im = (u+0)->im*(v+0)->re + (u+0)->re*(v+0)->im + (u+1)->im*(v+3)->re + (u+1)->re*(v+3)->im + (u+2)->im*(v+6)->re + (u+2)->re*(v+6)->im;
  (w+1)->re = (u+0)->re*(v+1)->re + (u+1)->re*(v+4)->re + (u+2)->re*(v+7)->re - (u+0)->im*(v+1)->im - (u+1)->im*(v+4)->im - (u+2)->im*(v+7)->im;
  (w+1)->im = (u+0)->im*(v+1)->re + (u+0)->re*(v+1)->im + (u+1)->im*(v+4)->re + (u+1)->re*(v+4)->im + (u+2)->im*(v+7)->re + (u+2)->re*(v+7)->im;
  (w+2)->re = (u+0)->re*(v+2)->re + (u+1)->re*(v+5)->re + (u+2)->re*(v+8)->re - (u+0)->im*(v+2)->im - (u+1)->im*(v+5)->im - (u+2)->im*(v+8)->im;
  (w+2)->im = (u+0)->im*(v+2)->re + (u+0)->re*(v+2)->im + (u+1)->im*(v+5)->re + (u+1)->re*(v+5)->im + (u+2)->im*(v+8)->re + (u+2)->re*(v+8)->im;

  (w+3)->re = (u+3)->re*(v+0)->re + (u+4)->re*(v+3)->re + (u+5)->re*(v+6)->re - (u+3)->im*(v+0)->im - (u+4)->im*(v+3)->im - (u+5)->im*(v+6)->im;
  (w+3)->im = (u+3)->im*(v+0)->re + (u+3)->re*(v+0)->im + (u+4)->im*(v+3)->re + (u+4)->re*(v+3)->im + (u+5)->im*(v+6)->re + (u+5)->re*(v+6)->im;
  (w+4)->re = (u+3)->re*(v+1)->re + (u+4)->re*(v+4)->re + (u+5)->re*(v+7)->re - (u+3)->im*(v+1)->im - (u+4)->im*(v+4)->im - (u+5)->im*(v+7)->im;
  (w+4)->im = (u+3)->im*(v+1)->re + (u+3)->re*(v+1)->im + (u+4)->im*(v+4)->re + (u+4)->re*(v+4)->im + (u+5)->im*(v+7)->re + (u+5)->re*(v+7)->im;
  (w+5)->re = (u+3)->re*(v+2)->re + (u+4)->re*(v+5)->re + (u+5)->re*(v+8)->re - (u+3)->im*(v+2)->im - (u+4)->im*(v+5)->im - (u+5)->im*(v+8)->im;
  (w+5)->im = (u+3)->im*(v+2)->re + (u+3)->re*(v+2)->im + (u+4)->im*(v+5)->re + (u+4)->re*(v+5)->im + (u+5)->im*(v+8)->re + (u+5)->re*(v+8)->im;

  (w+6)->re = (u+6)->re*(v+0)->re + (u+7)->re*(v+3)->re + (u+8)->re*(v+6)->re - (u+6)->im*(v+0)->im - (u+7)->im*(v+3)->im - (u+8)->im*(v+6)->im;
  (w+6)->im = (u+6)->im*(v+0)->re + (u+6)->re*(v+0)->im + (u+7)->im*(v+3)->re + (u+7)->re*(v+3)->im + (u+8)->im*(v+6)->re + (u+8)->re*(v+6)->im;
  (w+7)->re = (u+6)->re*(v+1)->re + (u+7)->re*(v+4)->re + (u+8)->re*(v+7)->re - (u+6)->im*(v+1)->im - (u+7)->im*(v+4)->im - (u+8)->im*(v+7)->im;
  (w+7)->im = (u+6)->im*(v+1)->re + (u+6)->re*(v+1)->im + (u+7)->im*(v+4)->re + (u+7)->re*(v+4)->im + (u+8)->im*(v+7)->re + (u+8)->re*(v+7)->im;
  (w+8)->re = (u+6)->re*(v+2)->re + (u+7)->re*(v+5)->re + (u+8)->re*(v+8)->re - (u+6)->im*(v+2)->im - (u+7)->im*(v+5)->im - (u+8)->im*(v+8)->im;
  (w+8)->im = (u+6)->im*(v+2)->re + (u+6)->re*(v+2)->im + (u+7)->im*(v+5)->re + (u+7)->re*(v+5)->im + (u+8)->im*(v+8)->re + (u+8)->re*(v+8)->im;
  return;
}


#endif /* _SU3_OPS_H */
