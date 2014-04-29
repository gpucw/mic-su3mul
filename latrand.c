#include <stdlib.h>
#include <string.h>
#include <utils.h>
#include <types.h>
#include <globals.h>
#include <su3_ops.h>
#include <math.h>

#ifndef M_PI
	#define M_PI 3.141592653589793238
#endif

void init_rand(int seed)
{
  srand(seed);
  return;
}

complex crand() 
{
  complex c;
  double pi = M_PI;
  double phi = 2*pi*rand()/(double)RAND_MAX;
  c.re = cos(phi);
  c.im = sin(phi);
  return c;
}

void exp_gell_mann(complex *u, double phi, int i_GM)
{
  switch(i_GM){
  case 1:
    (u+0)->re = cos(phi);
    (u+0)->im = 0.0;
    (u+1)->re = 0.0;
    (u+1)->im = sin(phi);
    (u+2)->re = 0.0;
    (u+2)->im = 0.0;

    (u+3)->re = 0.0;
    (u+3)->im = sin(phi);
    (u+4)->re = cos(phi);
    (u+4)->im = 0.0;
    (u+5)->re = 0.0;
    (u+5)->im = 0.0;

    (u+6)->re = 0.0;
    (u+6)->im = 0.0;
    (u+7)->re = 0.0;
    (u+7)->im = 0.0;
    (u+8)->re = 1.;
    (u+8)->im = 0.0;
    break;
  case 2:
    (u+0)->re = cos(phi);
    (u+0)->im = 0.0;
    (u+1)->re = sin(phi);
    (u+1)->im = 0.0;
    (u+2)->re = 0.0;
    (u+2)->im = 0.0;

    (u+3)->re =-sin(phi);
    (u+3)->im = 0.0;
    (u+4)->re = cos(phi);
    (u+4)->im = 0.0;
    (u+5)->re = 0.0;
    (u+5)->im = 0.0;

    (u+6)->re = 0.0;
    (u+6)->im = 0.0;
    (u+7)->re = 0.0;
    (u+7)->im = 0.0;
    (u+8)->re = 1.;
    (u+8)->im = 0.0;
    break;
  case 3:
    (u+0)->re = cos(phi);
    (u+0)->im = sin(phi);
    (u+1)->re = 0.0;
    (u+1)->im = 0.0;
    (u+2)->re = 0.0;
    (u+2)->im = 0.0;

    (u+3)->re = 0.0;
    (u+3)->im = 0.0;
    (u+4)->re = cos(phi);
    (u+4)->im =-sin(phi);
    (u+5)->re = 0.0;
    (u+5)->im = 0.0;

    (u+6)->re = 0.0;
    (u+6)->im = 0.0;
    (u+7)->re = 0.0;
    (u+7)->im = 0.0;
    (u+8)->re = 1.;
    (u+8)->im = 0.0;
    break;
  case 4:
    (u+0)->re = cos(phi);
    (u+0)->im = 0.0;
    (u+1)->re = 0.0;
    (u+1)->im = 0.0;
    (u+2)->re = 0.0;
    (u+2)->im = sin(phi);

    (u+3)->re = 0.0;
    (u+3)->im = 0.0;
    (u+4)->re = 1.;
    (u+4)->im = 0.0;
    (u+5)->re = 0.0;
    (u+5)->im = 0.0;

    (u+6)->re = 0.0;
    (u+6)->im = sin(phi);
    (u+7)->re = 0.0;
    (u+7)->im = 0.0;
    (u+8)->re = cos(phi);
    (u+8)->im = 0.0;
    break;
  case 5:
    (u+0)->re = cos(phi);
    (u+0)->im = 0.0;
    (u+1)->re = 0.0;
    (u+1)->im = 0.0;
    (u+2)->re = sin(phi);
    (u+2)->im = 0.0;

    (u+3)->re = 0.0;
    (u+3)->im = 0.0;
    (u+4)->re = 1.;
    (u+4)->im = 0.0;
    (u+5)->re = 0.0;
    (u+5)->im = 0.0;

    (u+6)->re =-sin(phi);
    (u+6)->im = 0.0;
    (u+7)->re = 0.0;
    (u+7)->im = 0.0;
    (u+8)->re = cos(phi);
    (u+8)->im = 0.0;
    break;
  case 6:
    (u+0)->re = 1.;
    (u+0)->im = 0.0;
    (u+1)->re = 0.0;
    (u+1)->im = 0.0;
    (u+2)->re = 0.0;
    (u+2)->im = 0.0;

    (u+3)->re = 0.0;
    (u+3)->im = 0.0;
    (u+4)->re = cos(phi);
    (u+4)->im = 0.0;
    (u+5)->re = 0.0;
    (u+5)->im = sin(phi);

    (u+6)->re = 0.0;
    (u+6)->im = 0.0;
    (u+7)->re = 0.0;
    (u+7)->im = sin(phi);
    (u+8)->re = cos(phi);
    (u+8)->im = 0.0;
    break;
  case 7:
    (u+0)->re = 1.;
    (u+0)->im = 0.0;
    (u+1)->re = 0.0;
    (u+1)->im = 0.0;
    (u+2)->re = 0.0;
    (u+2)->im = 0.0;

    (u+3)->re = 0.0;
    (u+3)->im = 0.0;
    (u+4)->re = cos(phi);
    (u+4)->im = 0.0;
    (u+5)->re = sin(phi);
    (u+5)->im = 0.0;

    (u+6)->re = 0.0;
    (u+6)->im = 0.0;
    (u+7)->re =-sin(phi);
    (u+7)->im = 0.0;
    (u+8)->re = cos(phi);
    (u+8)->im = 0.0;
    break;
  case 8:
    (u+0)->re = cos(phi/sqrt(3.));
    (u+0)->im = sin(phi/sqrt(3.));
    (u+1)->re = 0.0;
    (u+1)->im = 0.0;
    (u+2)->re = 0.0;
    (u+2)->im = 0.0;

    (u+3)->re = 0.0;
    (u+3)->im = 0.0;
    (u+4)->re = cos(phi/sqrt(3.));
    (u+4)->im = sin(phi/sqrt(3.));
    (u+5)->re = 0.0;
    (u+5)->im = 0.0;

    (u+6)->re = 0.0;
    (u+6)->im = 0.0;
    (u+7)->re = 0.0;
    (u+7)->im = 0.0;
    (u+8)->re = cos(phi*2./sqrt(3.));
    (u+8)->im =-sin(phi*2./sqrt(3.));
    break;
  }
}

void spinor_rand(complex *x)
{
  for(int v=0; v<length; v++)
    for(int s=0; s<NS*NC; s++) {
      x[v*NC*NS + s] = crand();
    }     
  return;
}

void su3_rand(complex *u)
{
  double amp = 1.0;
  for(int v=0; v<length; v++) {
    complex *uv = u+v*NC*NC;
    memset(uv, '\0', sizeof(complex)*NC*NC);
    uv[ 0].re = 1.0;
    uv[ 4].re = 1.0;
    uv[ 8].re = 1.0;
    for(int i=0; i<32; i++) {
      int igm = (int)(8.0*rand()/(double)RAND_MAX) + 1;
      complex aux1[NC*NC];
      complex aux2[NC*NC];
      double phi = M_PI*amp*((2.0*rand())/(double)RAND_MAX-1.0);
      exp_gell_mann(aux1, phi, igm);

      u_eq_u(aux2, uv);
      u_eq_uxu(uv, aux2, aux1);
    }
  }
  return;
}
