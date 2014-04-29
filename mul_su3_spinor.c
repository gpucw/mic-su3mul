#include <stdio.h>
#include <stdlib.h>
#include <utils.h>
#include <types.h>
#include <globals.h>
#include <string.h>

#ifdef SSE
	#include <xmmintrin.h>
#endif

#ifdef MIC
	#include <immintrin.h>
#endif

#ifdef OMP
	#include <omp.h>
#endif 

void mul_su3_spinor(complex *y, complex *u, complex *x)
{
	#ifdef OMP
		#pragma omp parallel for
	#endif
	
	for(int v=0; v<length; v++)
		for(int s=0; s<NS; s++)
			for(int c0=0; c0<NC; c0++) {
				complex y0 = (complex){0., 0.};
				complex *u0 = &u[v*NC*NC + c0*NC];
				complex *x0 = &x[v*NS*NC + s*NC];
				for(int c1=0; c1<NC; c1++) {
					y0.re += u0[c1].re*x0[c1].re - u0[c1].im*x0[c1].im;
					y0.im += u0[c1].re*x0[c1].im + u0[c1].im*x0[c1].re;
				}
				y[v*NS*NC + s*NC + c0] = y0;
			}
	return;
}

void mul_short_su3_spinor(complex *y, complex *u, complex *x)
{
	#ifdef OMP
		#pragma omp parallel for
	#endif
	for(int v=0; v<length; v++)
		for(int s=0; s<NS; s++) {
			
			/* c0 = 0 */
			int c0 = 0;
			complex y0 = (complex){0., 0.};
			complex *u0 = &u[v*(NC*NC-1) + c0*NC];
			complex *x0 = &x[v*NS*NC + s*NC];
			
			for(int c1=0; c1<NC; c1++) {
				y0.re += u0[c1].re*x0[c1].re - u0[c1].im*x0[c1].im;
				y0.im += u0[c1].re*x0[c1].im + u0[c1].im*x0[c1].re;
			}
			y[v*NS*NC + s*NC + c0] = y0;

			
			/* c0 = 1 */
			c0 = 1;
			y0 = (complex){0., 0.};
			complex *u1 = &u[v*(NC*NC-1) + c0*NC];
			
			for(int c1=0; c1<NC; c1++) {
				y0.re += u1[c1].re*x0[c1].re - u1[c1].im*x0[c1].im;
				y0.im += u1[c1].re*x0[c1].im + u1[c1].im*x0[c1].re;
			}
			y[v*NS*NC + s*NC + c0] = y0;

			
			/* c0 = 2 */
			c0 = 2;
			y0 = (complex){0., 0.};
			complex *u2 = &u[v*(NC*NC-1) + c0*NC];

			y0.re += u2[0].re*x0[0].re - u2[0].im*x0[0].im;
			y0.im += u2[0].re*x0[0].im + u2[0].im*x0[0].re;

			y0.re += u2[1].re*x0[1].re - u2[1].im*x0[1].im;
			y0.im += u2[1].re*x0[1].im + u2[1].im*x0[1].re;

			complex ux;
			ux.re = + u0[0].re*u1[1].re - u0[0].im*u1[1].im - u0[1].re*u1[0].re + u0[1].im*u1[0].im;
			ux.im = - u0[0].re*u1[1].im - u0[0].im*u1[1].re + u0[1].re*u1[0].im + u0[1].im*u1[0].re;

			y0.re += ux.re*x0[2].re - ux.im*x0[2].im;
			y0.im += ux.re*x0[2].im + ux.im*x0[2].re;

			y[v*NS*NC + s*NC + c0] = y0;      
		}

	return;
}

#ifdef SSE
void mul_su3_spinor_intrins(complex *y, complex *u, complex *x)
{
	#ifdef OMP
		#pragma omp parallel for
	#endif

	for(int i=0; i<length; i++) {
		double __attribute__((aligned(16))) s[2] = {-1.0, 1.0};
		__m128d sign = _mm_load_pd(s);
		__m128d register  g00 = _mm_load_pd(&u[0 + NC*NC*i].re);
		__m128d register  g01 = _mm_load_pd(&u[1 + NC*NC*i].re);
		__m128d register  g02 = _mm_load_pd(&u[2 + NC*NC*i].re);

		__m128d register  g10 = _mm_load_pd(&u[3 + NC*NC*i].re);
		__m128d register  g11 = _mm_load_pd(&u[4 + NC*NC*i].re);
		__m128d register  g12 = _mm_load_pd(&u[5 + NC*NC*i].re);
    
		__m128d register  g20 = _mm_load_pd(&u[6 + NC*NC*i].re);
		__m128d register  g21 = _mm_load_pd(&u[7 + NC*NC*i].re);
		__m128d register  g22 = _mm_load_pd(&u[8 + NC*NC*i].re);
    
		for(int sp=0; sp<NS; sp++) {
			__m128d register  x0 = _mm_load_pd(&x[0 + NC*sp + NC*NS*i].re);
			__m128d register  x1 = _mm_load_pd(&x[1 + NC*sp + NC*NS*i].re);
			__m128d register  x2 = _mm_load_pd(&x[2 + NC*sp + NC*NS*i].re);
		
			__m128d register  ix0 = _mm_shuffle_pd(x0, x0, _MM_SHUFFLE2(0, 1)); /* swap re with im */
			__m128d register  ix1 = _mm_shuffle_pd(x1, x1, _MM_SHUFFLE2(0, 1)); /* swap re with im */
			__m128d register  ix2 = _mm_shuffle_pd(x2, x2, _MM_SHUFFLE2(0, 1)); /* swap re with im */
      
			__m128d register  y0r, y0i, y0;
			__m128d register  y1r, y1i, y1;
			__m128d register  y2r, y2i, y2;
			__m128d register  a0, a1, a2, a3;
      
			y0r = _mm_mul_pd(g00, x0);
			a0 = _mm_mul_pd(g01, x1);
			y0r = _mm_add_pd(a0, y0r);
			a0 = _mm_mul_pd(g02, x2);
			y0r = _mm_add_pd(a0, y0r);
      
			y0i = _mm_mul_pd(g00, ix0);
			a0 = _mm_mul_pd(g01, ix1);
			y0i = _mm_add_pd(a0, y0i);
			a0 = _mm_mul_pd(g02, ix2);
			y0i = _mm_add_pd(a0, y0i);

			/* */
			y1r = _mm_mul_pd(g10, x0);
			a0 = _mm_mul_pd(g11, x1);
			y1r = _mm_add_pd(a0, y1r);
			a0 = _mm_mul_pd(g12, x2);
			y1r = _mm_add_pd(a0, y1r);

			y1i = _mm_mul_pd(g10, ix0);
			a0 = _mm_mul_pd(g11, ix1);
			y1i = _mm_add_pd(a0, y1i);
			a0 = _mm_mul_pd(g12, ix2);
			y1i = _mm_add_pd(a0, y1i);
          
			/* */
			y2r = _mm_mul_pd(g20, x0);
			a0 = _mm_mul_pd(g21, x1);
			y2r = _mm_add_pd(a0, y2r);
			a0 = _mm_mul_pd(g22, x2);
			y2r = _mm_add_pd(a0, y2r);
      
			y2i = _mm_mul_pd(g20, ix0);
			a0 = _mm_mul_pd(g21, ix1);
			y2i = _mm_add_pd(a0, y2i);
			a0 = _mm_mul_pd(g22, ix2);
			y2i = _mm_add_pd(a0, y2i);
      
			a0 = _mm_shuffle_pd(y0r, y0i, _MM_SHUFFLE2(0, 0));
			a1 = _mm_shuffle_pd(y0r, y0i, _MM_SHUFFLE2(1, 1));
			y0 = _mm_mul_pd(a1, sign);
			y0 = _mm_add_pd(a0, y0);

			a0 = _mm_shuffle_pd(y1r, y1i, _MM_SHUFFLE2(0, 0));
			a1 = _mm_shuffle_pd(y1r, y1i, _MM_SHUFFLE2(1, 1));
			y1 = _mm_mul_pd(a1, sign);
			y1 = _mm_add_pd(a0, y1);

			a0 = _mm_shuffle_pd(y2r, y2i, _MM_SHUFFLE2(0, 0));
			a1 = _mm_shuffle_pd(y2r, y2i, _MM_SHUFFLE2(1, 1));
			y2 = _mm_mul_pd(a1, sign);
			y2 = _mm_add_pd(a0, y2);

			_mm_stream_pd(&y[0 + NC*sp + NC*NS*i].re, y0);
			_mm_stream_pd(&y[1 + NC*sp + NC*NS*i].re, y1);      
			_mm_stream_pd(&y[2 + NC*sp + NC*NS*i].re, y2);      
		}
	}
	
	return;
}

void mul_short_su3_spinor_intrins(complex *y, complex *u, complex *x)
{
#ifdef OMP
#pragma omp parallel for
#endif
  for(int i=0; i<length; i++) {
    double __attribute__((aligned(16))) s[2] = {-1.0, 1.0};
    __m128d sign = _mm_load_pd(s);
    __m128d register  g00 = _mm_load_pd(&u[0 + (NC*NC-1)*i].re);
    __m128d register  g01 = _mm_load_pd(&u[1 + (NC*NC-1)*i].re);
    __m128d register  g02 = _mm_load_pd(&u[2 + (NC*NC-1)*i].re);
    
    __m128d register  g10 = _mm_load_pd(&u[3 + (NC*NC-1)*i].re);
    __m128d register  g11 = _mm_load_pd(&u[4 + (NC*NC-1)*i].re);
    __m128d register  g12 = _mm_load_pd(&u[5 + (NC*NC-1)*i].re);
    
    __m128d register  g20 = _mm_load_pd(&u[6 + (NC*NC-1)*i].re);
    __m128d register  g21 = _mm_load_pd(&u[7 + (NC*NC-1)*i].re);
    __m128d register  ig11 = _mm_shuffle_pd(g11, g11, _MM_SHUFFLE2(0, 1)); /* swap re with im */
    __m128d register  ig10 = _mm_shuffle_pd(g10, g10, _MM_SHUFFLE2(0, 1)); /* swap re with im */
    
    __m128d register g22_r, g22_i, g22, aux, a0, a1;
    
    g22_r = _mm_mul_pd(g00, g11);
    g22_r = _mm_mul_pd(sign, g22_r);
    
    aux = _mm_mul_pd(g01, g10);
    aux = _mm_mul_pd(sign, aux);

    g22_r = _mm_sub_pd(g22_r, aux);

    g22_i = _mm_mul_pd(g00, ig11);
    aux = _mm_mul_pd(g01, ig10);
    g22_i = _mm_sub_pd(aux, g22_i);

    a0 = _mm_shuffle_pd(g22_r, g22_i, _MM_SHUFFLE2(0, 0));
    a1 = _mm_shuffle_pd(g22_r, g22_i, _MM_SHUFFLE2(1, 1));

    g22 = _mm_add_pd(a0, a1);
    g22 = _mm_mul_pd(sign, g22);

    for(int sp=0; sp<NS; sp++) {
      __m128d register  x0 = _mm_load_pd(&x[0 + NC*sp + NC*NS*i].re);
      __m128d register  x1 = _mm_load_pd(&x[1 + NC*sp + NC*NS*i].re);
      __m128d register  x2 = _mm_load_pd(&x[2 + NC*sp + NC*NS*i].re);
      
      __m128d register  ix0 = _mm_shuffle_pd(x0, x0, _MM_SHUFFLE2(0, 1)); /* swap re with im */
      __m128d register  ix1 = _mm_shuffle_pd(x1, x1, _MM_SHUFFLE2(0, 1)); /* swap re with im */
      __m128d register  ix2 = _mm_shuffle_pd(x2, x2, _MM_SHUFFLE2(0, 1)); /* swap re with im */
      
      __m128d register  y0r, y0i, y0;
      __m128d register  y1r, y1i, y1;
      __m128d register  y2r, y2i, y2;

      y0r = _mm_mul_pd(g00, x0);
      a0 = _mm_mul_pd(g01, x1);
      y0r = _mm_add_pd(a0, y0r);
      a0 = _mm_mul_pd(g02, x2);
      y0r = _mm_add_pd(a0, y0r);
      
      y0i = _mm_mul_pd(g00, ix0);
      a0 = _mm_mul_pd(g01, ix1);
      y0i = _mm_add_pd(a0, y0i);
      a0 = _mm_mul_pd(g02, ix2);
      y0i = _mm_add_pd(a0, y0i);

      /* */
      y1r = _mm_mul_pd(g10, x0);
      a0 = _mm_mul_pd(g11, x1);
      y1r = _mm_add_pd(a0, y1r);
      a0 = _mm_mul_pd(g12, x2);
      y1r = _mm_add_pd(a0, y1r);

      y1i = _mm_mul_pd(g10, ix0);
      a0 = _mm_mul_pd(g11, ix1);
      y1i = _mm_add_pd(a0, y1i);
      a0 = _mm_mul_pd(g12, ix2);
      y1i = _mm_add_pd(a0, y1i);
          
      /* */
      y2r = _mm_mul_pd(g20, x0);
      a0 = _mm_mul_pd(g21, x1);
      y2r = _mm_add_pd(a0, y2r);
      a0 = _mm_mul_pd(g22, x2);
      y2r = _mm_add_pd(a0, y2r);
      
      y2i = _mm_mul_pd(g20, ix0);
      a0 = _mm_mul_pd(g21, ix1);
      y2i = _mm_add_pd(a0, y2i);
      a0 = _mm_mul_pd(g22, ix2);
      y2i = _mm_add_pd(a0, y2i);
      
      a0 = _mm_shuffle_pd(y0r, y0i, _MM_SHUFFLE2(0, 0));
      a1 = _mm_shuffle_pd(y0r, y0i, _MM_SHUFFLE2(1, 1));
      y0 = _mm_mul_pd(a1, sign);
      y0 = _mm_add_pd(a0, y0);

      a0 = _mm_shuffle_pd(y1r, y1i, _MM_SHUFFLE2(0, 0));
      a1 = _mm_shuffle_pd(y1r, y1i, _MM_SHUFFLE2(1, 1));
      y1 = _mm_mul_pd(a1, sign);
      y1 = _mm_add_pd(a0, y1);

      a0 = _mm_shuffle_pd(y2r, y2i, _MM_SHUFFLE2(0, 0));
      a1 = _mm_shuffle_pd(y2r, y2i, _MM_SHUFFLE2(1, 1));
      y2 = _mm_mul_pd(a1, sign);
      y2 = _mm_add_pd(a0, y2);

      _mm_stream_pd(&y[0 + NC*sp + NC*NS*i].re, y0);
      _mm_stream_pd(&y[1 + NC*sp + NC*NS*i].re, y1);      
      _mm_stream_pd(&y[2 + NC*sp + NC*NS*i].re, y2);      
    }
  }
  return;
}
#endif

#ifdef MIC
void mul_su3_spinor_intrins(complex *y, complex *u, complex *x)
{
	double __attribute__((aligned(128))) s[8] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
	__m512d sign = _mm512_load_pd(s);

	#ifdef OMP
		#pragma omp parallel for
	#endif

	for (int l=0; l<length; l++)
	{
		for (int sp=0; sp<NS; sp++)
		{
			__m512d rpy = _mm512_load_pd(y + l*NS*NS + sp*NS);
			
			__m512d rpx = _mm512_load_pd(x + l*NS*NS + sp*NS);
			__m512d rpx_i = _mm512_swizzle_pd(rpx, _MM_SWIZ_REG_CDAB);
	
			__m512d rpu0 = _mm512_load_pd(u + l*NC*NS);
			__m512d rpu1 = _mm512_load_pd(u + l*NC*NS + 4);
			__m512d rpu2 = _mm512_load_pd(u + l*NC*NS + 8);
	
			__m512d a0, a1;
	
			////// 0 //////
			__m512d r0r = _mm512_mul_pd(rpx, rpu0);
			__m512d r0i = _mm512_mul_pd(rpx_i, rpu0);
			a0 = _mm512_mask_swizzle_pd(r0r, 0xaa, r0i, _MM_SWIZ_REG_CDAB);
			a1 = _mm512_mask_swizzle_pd(r0i, 0x55, r0r, _MM_SWIZ_REG_CDAB);
			r0r = _mm512_mul_pd(a1, sign);
			__m512d r0 = _mm512_add_pd(r0r, a0);
			double r0RE = _mm512_mask_reduce_add_pd(0x55,r0);
			double r0IM = _mm512_mask_reduce_add_pd(0xaa,r0);
	
			////// 1 //////
			__m512d r1r = _mm512_mul_pd(rpx, rpu1);
			__m512d r1i = _mm512_mul_pd(rpx_i, rpu1);
			a0 = _mm512_mask_swizzle_pd(r1r, 0xaa, r1i, _MM_SWIZ_REG_CDAB);
			a1 = _mm512_mask_swizzle_pd(r1i, 0x55, r1r, _MM_SWIZ_REG_CDAB);
			r1r = _mm512_mul_pd(a1, sign);
			__m512d r1 = _mm512_add_pd(r1r, a0);
			double r1RE = _mm512_mask_reduce_add_pd(0x55,r1);
			double r1IM = _mm512_mask_reduce_add_pd(0xaa,r1);
	
			////// 2 //////
			__m512d r2r = _mm512_mul_pd(rpx, rpu2);
			__m512d r2i = _mm512_mul_pd(rpx_i, rpu2);
			a0 = _mm512_mask_swizzle_pd(r2r, 0xaa, r2i, _MM_SWIZ_REG_CDAB);
			a1 = _mm512_mask_swizzle_pd(r2i, 0x55, r2r, _MM_SWIZ_REG_CDAB);
			r2r = _mm512_mul_pd(a1, sign);
			__m512d r2 = _mm512_add_pd(r2r, a0);
			double r2RE = _mm512_mask_reduce_add_pd(0x55,r2);
			double r2IM = _mm512_mask_reduce_add_pd(0xaa,r2);
	
		
			double __attribute__((aligned(128))) temp[8] = {r0RE, r0IM, r1RE, r1IM, r2RE, r2IM, 0.0, 0.0};
			rpy = _mm512_load_pd(temp);
		
			_mm512_store_pd((void *)(y + l*NS*NS + sp*NS), rpy);
		
		}
	}
}

void mul_short_su3_spinor_intrins(complex *y, complex *u, complex *x)
{
#ifdef OMP
#pragma omp parallel for
#endif
  for(int v=0; v<length; v++) {
    __m512d ru0 = _mm512_load_pd((double *)&u[     v*8].re);
    __m512d ru1 = _mm512_load_pd((double *)&u[ 4 + v*8].re);
    __m512d ru2 = _mm512_load_pd((double *)&u[ 8 + v*8].re);
    __m512i perm = {0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,};
    /* __m512d ru0 = (__m512d) _mm512_permutevar_epi32(perm, (__m512i)ru); */
    /* //__m512d rpx0i = _mm512_swizzle_pd(rpx0, _MM_SWIZ_REG_CDAB);       */
    complex aux[4] __attribute__ ((aligned(128)));
    _mm512_store_pd((double *)aux, ru0);
    //printf("1 %+e%+e\n", aux[0].re, aux[0].im);
    //printf("1 %+e%+e\n", aux[1].re, aux[1].im);
    //printf("1 %+e%+e\n", aux[2].re, aux[2].im);
    //printf("1 %+e%+e\n", aux[3].re, aux[3].im);
  }
  return;
}
#endif

#ifdef REOR
void mul_su3_spinor_reor(complex *y, complex *u, complex *x)
{
	int ithr = 0;
	int nthr = 1;
	#ifdef OMP
		#pragma omp parallel
		{
			nthr = omp_get_num_threads();
			ithr = omp_get_thread_num();
	#endif
    
			int thr_len = length / nthr;
			complex *y0 = &y[(thr_len*ithr) * NC*NS];
			complex *x0 = &x[(thr_len*ithr) * NC*NS];
			complex *u0 = &u[(thr_len*ithr) * NC*NC];
			memset(y0, '\0', thr_len*NC*NS*sizeof(complex));

			for(int sp=0; sp<NS; sp++)
				for(int c0=0; c0<NC; c0++) 
				{
					complex *py = &y0[thr_len*(c0 + sp*NC)];
					for(int c1=0; c1<NC; c1++) 
					{
						complex *px = &x0[thr_len*(c1 + sp*NC)];
						complex *pu = &u0[thr_len*(c1 + c0*NC)];
						for(int v=0; v<thr_len; v++) 
						{
							py[v].re += px[v].re * pu[v].re - px[v].im * pu[v].im;
							py[v].im += px[v].im * pu[v].re + px[v].re * pu[v].im;
						}
					}
				}
	
#ifdef OMP
		}
#endif
  return;
}
#endif
 
#if defined(REOR) & defined(SSE) 
void mul_su3_spinor_reor_intrins(complex *y, complex *u, complex *x)
{
	int ithr = 0;
	int nthr = 1;
	
	#ifdef OMP
		#pragma omp parallel
		{
			nthr = omp_get_num_threads();
			ithr = omp_get_thread_num();
	#endif
    
    int thr_len = length / nthr;
    complex *y0 = &y[(thr_len*ithr) * NC*NS];
    complex *x0 = &x[(thr_len*ithr) * NC*NS];
    complex *u0 = &u[(thr_len*ithr) * NC*NC];
    memset(y0, '\0', thr_len*NC*NS*sizeof(complex));
    
    double __attribute__((aligned(16))) s[2] = {-1.0, 1.0};
    __m128d sign = _mm_load_pd(s);
    
    for(int sp=0; sp<NS; sp++)
		for(int c0=0; c0<NC; c0++) 
		{
			complex *py = &y0[thr_len*(c0 + sp*NC)];
			for(int c1=0; c1<NC; c1++) 
			{
				complex *px = &x0[thr_len*(c1 + sp*NC)];
				complex *pu = &u0[thr_len*(c1 + c0*NC)];
				for(int v=0; v<thr_len; v+=4) 
				{
					__m128d rpy0 = _mm_load_pd((double *)(py+v+0));
					__m128d rpy1 = _mm_load_pd((double *)(py+v+1));
					__m128d rpy2 = _mm_load_pd((double *)(py+v+2));
					__m128d rpy3 = _mm_load_pd((double *)(py+v+3));
					__m128d rpx0 = _mm_load_pd((double *)(px+v+0));
					__m128d rpx1 = _mm_load_pd((double *)(px+v+1));
					__m128d rpx2 = _mm_load_pd((double *)(px+v+2));
					__m128d rpx3 = _mm_load_pd((double *)(px+v+3));
					__m128d rpx0i = _mm_shuffle_pd(rpx0, rpx0, _MM_SHUFFLE2(0, 1)); /* swap re with im */
					__m128d rpx1i = _mm_shuffle_pd(rpx1, rpx1, _MM_SHUFFLE2(0, 1)); /* swap re with im */
					__m128d rpx2i = _mm_shuffle_pd(rpx2, rpx2, _MM_SHUFFLE2(0, 1)); /* swap re with im */
					__m128d rpx3i = _mm_shuffle_pd(rpx3, rpx3, _MM_SHUFFLE2(0, 1)); /* swap re with im */
					__m128d rpu0 = _mm_load_pd((double *)(pu+v+0));
					__m128d rpu1 = _mm_load_pd((double *)(pu+v+1));
					__m128d rpu2 = _mm_load_pd((double *)(pu+v+2));
					__m128d rpu3 = _mm_load_pd((double *)(pu+v+3));
					__m128d r0r, r0i;
					__m128d r1r, r1i;
					__m128d r2r, r2i;
					__m128d r3r, r3i;
					__m128d a0, a1;
		
					/*
					py[v].re += 
					+ px[v].re * pu[v].re 
					- px[v].im * pu[v].im;
					py[v].im += 
					+ px[v].im * pu[v].re 
					+ px[v].re * pu[v].im;
					*/
					r0r = _mm_mul_pd(rpx0, rpu0);
					r0i = _mm_mul_pd(rpx0i, rpu0);

					r1r = _mm_mul_pd(rpx1, rpu1);
					r1i = _mm_mul_pd(rpx1i, rpu1);

					r2r = _mm_mul_pd(rpx2, rpu2);
					r2i = _mm_mul_pd(rpx2i, rpu2);

					r3r = _mm_mul_pd(rpx3, rpu3);
					r3i = _mm_mul_pd(rpx3i, rpu3);

					/* 0 */
					a0 = _mm_shuffle_pd(r0r, r0i, _MM_SHUFFLE2(0, 0));
					a1 = _mm_shuffle_pd(r0r, r0i, _MM_SHUFFLE2(1, 1));
					rpy0 = _mm_add_pd(a0, rpy0);
					a0 = _mm_mul_pd(sign, a1);
					rpy0 = _mm_add_pd(a0, rpy0);

					/* 1 */
					a0 = _mm_shuffle_pd(r1r, r1i, _MM_SHUFFLE2(0, 0));
					a1 = _mm_shuffle_pd(r1r, r1i, _MM_SHUFFLE2(1, 1));
					rpy1 = _mm_add_pd(a0, rpy1);
					a0 = _mm_mul_pd(sign, a1);
					rpy1 = _mm_add_pd(a0, rpy1);

					/* 2 */
					a0 = _mm_shuffle_pd(r2r, r2i, _MM_SHUFFLE2(0, 0));
					a1 = _mm_shuffle_pd(r2r, r2i, _MM_SHUFFLE2(1, 1));
					rpy2 = _mm_add_pd(a0, rpy2);
					a0 = _mm_mul_pd(sign, a1);
					rpy2 = _mm_add_pd(a0, rpy2);

					/* 3 */
					a0 = _mm_shuffle_pd(r3r, r3i, _MM_SHUFFLE2(0, 0));
					a1 = _mm_shuffle_pd(r3r, r3i, _MM_SHUFFLE2(1, 1));
					rpy3 = _mm_add_pd(a0, rpy3);
					a0 = _mm_mul_pd(sign, a1);
					rpy3 = _mm_add_pd(a0, rpy3);

					_mm_store_pd((double *)(py+v+0), rpy0);
					_mm_store_pd((double *)(py+v+1), rpy1);      
					_mm_store_pd((double *)(py+v+2), rpy2);      
					_mm_store_pd((double *)(py+v+3), rpy3);      
				}
			}
		}
    
	#ifdef OMP
	}
	#endif
	
	return;
}
#endif

#if defined(REOR) & defined(MIC) 
void mul_su3_spinor_reor_intrins(complex *y, complex *u, complex *x)
{
  int ithr = 0;
  int nthr = 1;
  
  #ifdef OMP
  #pragma omp parallel
  {
    nthr = omp_get_num_threads();
    ithr = omp_get_thread_num();
  #endif
  
    int thr_len = length / nthr;
    complex *y0 = &y[(thr_len*ithr) * NC*NS];
    complex *x0 = &x[(thr_len*ithr) * NC*NS];
    complex *u0 = &u[(thr_len*ithr) * NC*NC];
    memset(y0, '\0', thr_len*NC*NS*sizeof(complex));
    double __attribute__((aligned(128))) s[8] = {-1.0, 1.0, 
						 -1.0, 1.0, 
						 -1.0, 1.0, 
						 -1.0, 1.0};
    __m512d sign = _mm512_load_pd(s);
    for(int sp=0; sp<NS; sp++)
      for(int c0=0; c0<NC; c0++) {
    	complex *py = &y0[thr_len*(c0 + sp*NC)];
    	for(int c1=0; c1<NC; c1++) {
    	  complex *px = &x0[thr_len*(c1 + sp*NC)];
    	  complex *pu = &u0[thr_len*(c1 + c0*NC)];
    	  for(int v=0; v<thr_len; v+=16) {
	    __m512d rpy0 = _mm512_load_pd((double *)(py+v+ 0));
	    __m512d rpy1 = _mm512_load_pd((double *)(py+v+ 4));
	    __m512d rpy2 = _mm512_load_pd((double *)(py+v+ 8));
	    __m512d rpy3 = _mm512_load_pd((double *)(py+v+12));

	    __m512d rpx0 = _mm512_load_pd((double *)(px+v+ 0));
	    __m512d rpx1 = _mm512_load_pd((double *)(px+v+ 4));
	    __m512d rpx2 = _mm512_load_pd((double *)(px+v+ 8));
	    __m512d rpx3 = _mm512_load_pd((double *)(px+v+12));

	    __m512d rpx0i = _mm512_swizzle_pd(rpx0, _MM_SWIZ_REG_CDAB); /* swap re with im */
	    __m512d rpx1i = _mm512_swizzle_pd(rpx1, _MM_SWIZ_REG_CDAB); /* swap re with im */
	    __m512d rpx2i = _mm512_swizzle_pd(rpx2, _MM_SWIZ_REG_CDAB); /* swap re with im */
	    __m512d rpx3i = _mm512_swizzle_pd(rpx3, _MM_SWIZ_REG_CDAB); /* swap re with im */

	    __m512d rpu0 = _mm512_load_pd((double *)(pu+v+ 0));
	    __m512d rpu1 = _mm512_load_pd((double *)(pu+v+ 4));
	    __m512d rpu2 = _mm512_load_pd((double *)(pu+v+ 8));
	    __m512d rpu3 = _mm512_load_pd((double *)(pu+v+12));


	    __m512d r0r, r0i;
	    __m512d r1r, r1i;
	    __m512d r2r, r2i;
	    __m512d r3r, r3i;


	    __m512d a0, a1;
	    r0r = _mm512_mul_pd(rpx0, rpu0); 
	    r0i = _mm512_mul_pd(rpx0i, rpu0);
	    r1r = _mm512_mul_pd(rpx1, rpu1); 
	    r1i = _mm512_mul_pd(rpx1i, rpu1);
	    r2r = _mm512_mul_pd(rpx2, rpu2); 
	    r2i = _mm512_mul_pd(rpx2i, rpu2);
	    r3r = _mm512_mul_pd(rpx3, rpu3); 
	    r3i = _mm512_mul_pd(rpx3i, rpu3);

    	    a0 = _mm512_mask_swizzle_pd(r0r, 0xaa, r0i, _MM_SWIZ_REG_CDAB);
    	    a1 = _mm512_mask_swizzle_pd(r0i, 0x55, r0r, _MM_SWIZ_REG_CDAB);
	    
	    r0r = _mm512_fmadd_pd(a1, sign, rpy0);
    	    rpy0 = _mm512_add_pd(r0r, a0);

    	    a0 = _mm512_mask_swizzle_pd(r1r, 0xaa, r1i, _MM_SWIZ_REG_CDAB);
    	    a1 = _mm512_mask_swizzle_pd(r1i, 0x55, r1r, _MM_SWIZ_REG_CDAB);
	    
	    r1r = _mm512_fmadd_pd(a1, sign, rpy1);
    	    rpy1 = _mm512_add_pd(r1r, a0);

    	    a0 = _mm512_mask_swizzle_pd(r2r, 0xaa, r2i, _MM_SWIZ_REG_CDAB);
    	    a1 = _mm512_mask_swizzle_pd(r2i, 0x55, r2r, _MM_SWIZ_REG_CDAB);
	    
	    r2r = _mm512_fmadd_pd(a1, sign, rpy2);
    	    rpy2 = _mm512_add_pd(r2r, a0);

    	    a0 = _mm512_mask_swizzle_pd(r3r, 0xaa, r3i, _MM_SWIZ_REG_CDAB);
    	    a1 = _mm512_mask_swizzle_pd(r3i, 0x55, r3r, _MM_SWIZ_REG_CDAB);
	    
	    r3r = _mm512_fmadd_pd(a1, sign, rpy3);
    	    rpy3 = _mm512_add_pd(r3r, a0);
	    /*
    	    py[v].re +=
    	      + px[v].re * pu[v].re
    	      - px[v].im * pu[v].im;
    	    py[v].im +=
    	      + px[v].im * pu[v].re
    	      + px[v].re * pu[v].im;
	    */

    	    _mm512_store_pd((void *)(py+v+ 0), rpy0);
    	    _mm512_store_pd((void *)(py+v+ 4), rpy1);
    	    _mm512_store_pd((void *)(py+v+ 8), rpy2);
    	    _mm512_store_pd((void *)(py+v+12), rpy3);
	  }
	}
      }
#ifdef OMP
  }
#endif
  return;
}

void mul_su3_spinor_hopping_reor_intrins(complex *y, complex *u, complex *x)
{	
	int ithr = 0;
	int nthr = 1;
	
	#ifdef OMP
		#pragma omp parallel
		{
			nthr = omp_get_num_threads();
		}
	#endif
    
    int thr_len = length / nthr;
	
	double __attribute__((aligned(128))) s[8] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
	__m512d sign = _mm512_load_pd(s);
	
	#ifdef OMP
		#pragma omp parallel for
	#endif
	
	
	for(int ithr=0; ithr<nthr; ithr++)
		for (int v=0; v<thr_len/4; v++)
			for (int sp=0; sp<NS; sp++)
				for (int co=0; co<NC; co++)
				{
					__m512d rsum = _mm512_set1_pd(0.0);
			
					for (int k=0; k<NC; k++)
					{
						__m512d r_x = _mm512_load_pd(&x[4*(k + NC*(sp + NS*(v + thr_len/4*ithr)))]);
						__m512d r_u = _mm512_load_pd(&u[4*(k + NC*(co + NC*(v + thr_len/4*ithr)))]);
			
						// calculate real elements
						__m512d real = _mm512_mul_pd(r_x, r_u);
						__m512d a0 = _mm512_swizzle_pd(real, _MM_SWIZ_REG_CDAB);
						a0 = _mm512_mul_pd(a0, sign);
						real = _mm512_add_pd(real, a0);
	
						//calculate imaginary elements
						a0 = _mm512_swizzle_pd(r_x, _MM_SWIZ_REG_CDAB);
						__m512d imag = _mm512_mul_pd(a0, r_u);
						__m512d a1 = _mm512_swizzle_pd(imag, _MM_SWIZ_REG_CDAB);
						imag = _mm512_add_pd(imag, a1);
	
						__m512d res = _mm512_mask_swizzle_pd(real, 0xaa, imag, _MM_SWIZ_REG_CDAB);
						rsum = _mm512_add_pd(rsum, res);
					}
					
					_mm512_store_pd(&y[4*(co + NC*(sp + NS*(v + thr_len/4*ithr)))], rsum);
				}
	return;
}
#endif







