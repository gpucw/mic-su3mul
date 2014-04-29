#include <stdlib.h>
#include <utils.h>
#include <types.h>
#include <globals.h>

#ifdef OMP
	#include <omp.h>
#endif

void spinor_conv(complex *out, complex *in, enum conv c)
{
	int nthr = 1;
	int i = 0;
	
	#ifdef OMP
		#pragma omp parallel
		{
			nthr = omp_get_num_threads();
		}
	#endif
	int thr_len = length / nthr;
  
	switch(c) {
		case XSLOW_TO_XFAST:
			for(int ithr=0; ithr<nthr; ithr++)
				for(int x=0; x<thr_len; x++)
					for(int sp=0; sp<NS; sp++)
						for(int co=0; co<NC; co++)
							out[x + thr_len*(co + NC*(sp + NS*ithr))] = in[co + NC*(sp + NS*(x + thr_len*ithr))];
		break;
		
		case XFAST_TO_XSLOW:
			for(int ithr=0; ithr<nthr; ithr++)
				for(int x=0; x<thr_len; x++)
					for(int sp=0; sp<NS; sp++)
						for(int co=0; co<NC; co++)
							out[co + NC*(sp + NS*(x + thr_len*ithr))] = in[x + thr_len*(co + NC*(sp + NS*ithr))];
		break;
		
		case HOPPING_XSLOW_TO_XFAST:
			i = 0;
			for(int ithr=0; ithr<nthr; ithr++)
				for(int x=0; x<thr_len/4; x++)
					for(int sp=0; sp<NS; sp++)
						for(int co=0; co<NC; co++)
						{
							out[i] = in[co + NC*(sp + NS*((x+0) + thr_len*ithr))];
							out[i+1] = in[co + NC*(sp + NS*((x+1*thr_len/4) + thr_len*ithr))];
							out[i+2] = in[co + NC*(sp + NS*((x+2*thr_len/4) + thr_len*ithr))];
							out[i+3] = in[co + NC*(sp + NS*((x+3*thr_len/4) + thr_len*ithr))];
							i+=4;
						}
		break;
		
		case HOPPING_XFAST_TO_XSLOW:
			i = 0;
			for(int ithr=0; ithr<nthr; ithr++)
				for(int x=0; x<thr_len/4; x++)
					for(int sp=0; sp<NS; sp++)
						for(int co=0; co<NC; co++)
						{
							out[co + NC*(sp + NS*((x+0) + thr_len*ithr))] = in[i];
							out[co + NC*(sp + NS*((x+1*thr_len/4) + thr_len*ithr))] = in[i + 1];
							out[co + NC*(sp + NS*((x+2*thr_len/4) + thr_len*ithr))] = in[i + 2];
							out[co + NC*(sp + NS*((x+3*thr_len/4) + thr_len*ithr))] = in[i + 3];
							i+=4;
						}
			break;
	}

	return;
}

void su3_conv(complex *out, complex *in, enum conv c)
{
	int nthr = 1;
	int i = 0;
	
	#ifdef OMP
		#pragma omp parallel
		{
			nthr = omp_get_num_threads();
		}
	#endif
	int thr_len = length / nthr;
  
	switch(c) {
		case XSLOW_TO_XFAST:
			for(int ithr=0; ithr<nthr; ithr++)
				for(int x=0; x<thr_len; x++)
					for(int c0=0; c0<NC; c0++)
						for(int c1=0; c1<NC; c1++)
							out[x + thr_len*(c1 + NC*(c0 + NC*ithr))] = in[c1 + NC*(c0 + NC*(x + thr_len*ithr))];
		break;
		
		case XFAST_TO_XSLOW:
			for(int ithr=0; ithr<nthr; ithr++)
				for(int x=0; x<thr_len; x++)
					for(int c0=0; c0<NC; c0++)
						for(int c1=0; c1<NC; c1++)
			out[c1 + NC*(c0 + NC*(x + thr_len*ithr))] = in[x + thr_len*(c1 + NC*(c0 + NC*ithr))];
		break;
		
		case HOPPING_XSLOW_TO_XFAST:
			i = 0;
			for(int ithr=0; ithr<nthr; ithr++)
				for(int x=0; x<thr_len/4; x++)
					for(int c0=0; c0<NC; c0++)
						for(int c1=0; c1<NC; c1++)
						{
							out[i] = in[c1 + NC*(c0 + NC*((x+0) + thr_len*ithr))];
							out[i+1] = in[c1 + NC*(c0 + NC*((x+1*thr_len/4) + thr_len*ithr))];
							out[i+2] = in[c1 + NC*(c0 + NC*((x+2*thr_len/4) + thr_len*ithr))];
							out[i+3] = in[c1 + NC*(c0 + NC*((x+3*thr_len/4) + thr_len*ithr))];
							i+=4;
						}
		break;
		
		case HOPPING_XFAST_TO_XSLOW:
			i = 0;
			for(int ithr=0; ithr<nthr; ithr++)
				for(int x=0; x<thr_len/4; x++)
					for(int c0=0; c0<NC; c0++)
						for(int c1=0; c1<NC; c1++)
						{
							out[c1 + NC*(c0 + NC*((x+0) + thr_len*ithr))] = in[i];
							out[c1 + NC*(c0 + NC*((x+1*thr_len/4) + thr_len*ithr))] = in[i + 1];
							out[c1 + NC*(c0 + NC*((x+2*thr_len/4) + thr_len*ithr))] = in[i + 2];
							out[c1 + NC*(c0 + NC*((x+3*thr_len/4) + thr_len*ithr))] = in[i + 3];
							i+=4;
						}
		break;
	}

	return;
}

void su3_to_short_su3(complex *out, complex *in)
{
	for(int x=0; x<length; x++)
		for(int co=0; co<NC*NC-1; co++)
			out[co + (NC*NC-1)*x] = in[co + NC*NC*x];
    
  return;
}

void copy4x3_to_4x4(complex* m4x3, complex* m4x4)
{
	for (int l=0; l<length; l++)
	{
		for (int i=0; i<NS; i++)
			for (int j=0; j<NC; j++)
				m4x4[l*NS*NS + i*NS + j] = m4x3[l*NS*NC + i*NC + j];
			
		m4x4[l*NS*NS + 3].re=0.0;
		m4x4[l*NS*NS + 3].im=0.0;
		m4x4[l*NS*NS + 7].re=0.0;
		m4x4[l*NS*NS + 7].im=0.0;
		m4x4[l*NS*NS + 11].re=0.0;
		m4x4[l*NS*NS + 11].im=0.0;
		m4x4[l*NS*NS + 15].re=0.0;
		m4x4[l*NS*NS + 15].im=0.0;
	}
}

void copy3x3_to_3x4(complex* m3x3, complex* m3x4)
{
	
	for (int l=0; l<length; l++)
	{
		for (int i=0; i<NC; i++)
			for (int j=0; j<NC; j++)
				m3x4[l*NC*NS + i*NS + j] = m3x3[l*NC*NC + i*NC + j];
	
		m3x4[l*NC*NS + 3].re=0.0;
		m3x4[l*NC*NS + 3].im=0.0;
		m3x4[l*NC*NS + 7].re=0.0;
		m3x4[l*NC*NS + 7].im=0.0;
		m3x4[l*NC*NS + 11].re=0.0;
		m3x4[l*NC*NS + 11].im=0.0;
	}
}

void copy4x4_to_4x3(complex* m4x4, complex* m4x3)
{
	for (int l=0; l<length; l++)
		for (int i=0; i<NS; i++)
			for (int j=0; j<NC; j++)
				m4x3[l*NS*NC + i*NC + j] = m4x4[l*NS*NS + i*NS + j];
}
