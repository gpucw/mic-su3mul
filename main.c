#include <stdio.h>
#include <stdlib.h>
#include <types.h>
#include <globals.h>
#include <utils.h>
#include <latinit.h>
#include <latrand.h>
#include <latconv.h>
#include <mul_su3_spinor.h>
#include <string.h>

#ifdef OMP
	#include <omp.h>
#endif

#define SEED 1771

void usage(char *argv[])
{
	fprintf(stderr, " Usage: %s LENGTH NITERS\n", argv[0]);
	return;
}

double diff(complex *x, complex *y)
{
	double sum = 0;
	double nrm = 0;
	for(int i=0; i<NS*NC*length; i++) {
		double dr = x[i].re - y[i].re;
		double di = x[i].im - y[i].im;
		double sr = x[i].re + y[i].re;
		double si = x[i].im + y[i].im;
		sum += (dr*dr + di*di);
		nrm += (sr*sr + si*si);
	}
  
	return sum/nrm;
}

int main(int argc, char *argv[])
{
	if(argc != 3) 
	{
		usage(argv);
		exit(1);
	}

	int nthr = 1;

	#ifdef OMP
		#pragma omp parallel
		{
			nthr = omp_get_num_threads();
		}
	#endif

	length = atol(argv[1]);
	unsigned long int niters = atol(argv[2]);

	complex *x = spinor_init(length);
	complex *y = spinor_init(length);
	complex *u = su3_init(length); 

	init_rand(SEED);
	spinor_rand(x);
	su3_rand(u);
	
	complex *u_sh = su3_short_init(length);
	su3_to_short_su3(u_sh, u);

	mul_su3_spinor(y, u, x);
	double t = stop_watch(0);
	for(int i=0; i<niters; i++)
		mul_su3_spinor(y, u, x);
	t = stop_watch(t);

	printf(" 0: nthr = %6d, %8lu iters, length = %8lu, t = %10.2lf usec per iter, BW = %6.2lf GB/s, perf = %12.2lf Mflop/s\n", nthr, niters, length, t/(double)niters, (2*NS*NC+NC*NC)*sizeof(complex)*length*niters/t*1e6/1024/1024/1024, 66*(double)niters*NS*(double)length/t);
	double d = 0;
	
	
	complex *z = spinor_init(length);
	mul_short_su3_spinor(z, u_sh, x);
	t = stop_watch(0);
	for(int i=0; i<niters; i++)
		mul_short_su3_spinor(z, u_sh, x);
	t = stop_watch(t);
  
	d = diff(y,z);  
	printf(" 1: nthr = %6d, %8lu iters, length = %8lu, t = %10.2lf usec per iter, BW = %6.2lf GB/s, perf = %12.2lf Mflop/s, diff = %e\n", nthr, niters, length, t/(double)niters, (2*NS*NC+NC*NC-1)*sizeof(complex)*length*niters/t*1e6/1024/1024/1024, 66*(double)niters*NS*(double)length/t, d);
	
	#if defined(SSE)
		mul_su3_spinor_intrins(z, u, x);
		t = stop_watch(0);
		for(int i=0; i<niters; i++)
			mul_su3_spinor_intrins(z, u, x);
		t = stop_watch(t);

		d = diff(y,z);
  
		printf(" 2: nthr = %6d, %8lu iters, length = %8lu, t = %10.2lf usec per iter, BW = %6.2lf GB/s, perf = %12.2lf Mflop/s, diff = %e\n", nthr, niters, length, t/(double)niters, (2*NS*NC+NC*NC)*sizeof(complex)*length*niters/t*1e6/1024/1024/1024, 66*(double)niters*NS*(double)length/t, d);
	#endif

	//code segment added for MIC intrins
	#if defined(MIC)
		complex* xx = alloc(length*NS*NS*sizeof(complex)); 		// xx => 4x4 
		complex* uu = alloc(length*NC*NS*sizeof(complex));		// uu => 3x4
		complex* yy = alloc(length*NS*NS*sizeof(complex));		// yy => 4x4
	
		memset(xx, '\0', length*NS*NS*sizeof(complex));
		memset(uu, '\0', length*NC*NS*sizeof(complex));
		memset(yy, '\0', length*NS*NS*sizeof(complex));

		copy4x3_to_4x4(x, xx);
		copy3x3_to_3x4(u, uu);

		mul_su3_spinor_intrins(yy, uu, xx);
		t = stop_watch(0);
		
		for (int i=0; i<niters; i++)
			mul_su3_spinor_intrins(yy, uu, xx);
		
		t = stop_watch(t);	
		copy4x4_to_4x3(yy, z);
		d = diff(y, z);
		
		printf(" 2: nthr = %6d, %8lu iters, length = %8lu, t = %10.2lf usec per iter, BW = %6.2lf GB/s, perf = %12.2lf Mflop/s, diff = %e\n", nthr, niters, length, t/(double)niters, (2*NS*NC+NC*NC)*sizeof(complex)*length*niters/t*1e6/1024/1024/1024, 66*(double)niters*NS*(double)length/t, d);
	
		free(xx);
		free(uu);
		free(yy);
	#endif


	memset(z, '\0', length*NC*NS*sizeof(complex)); //this line shows that short_intrins multiply operation for MIC is false
	#if defined(SSE)||defined(MIC)
		mul_short_su3_spinor_intrins(z, u_sh, x);
		t = stop_watch(0);
		for(int i=0; i<niters; i++)
			mul_short_su3_spinor_intrins(z, u_sh, x);
		t = stop_watch(t);

		d = diff(y,z);
		printf(" 3: nthr = %6d, %8lu iters, length = %8lu, t = %10.2lf usec per iter, BW = %6.2lf GB/s, perf = %12.2lf Mflop/s, diff = %e\n", nthr, niters, length, t/(double)niters, (2*NS*NC+NC*NC-1)*sizeof(complex)*length*niters/t*1e6/1024/1024/1024, 66*(double)niters*NS*(double)length/t, d);
		
		free(u_sh);
		free(z);
	#endif

	#ifdef REOR
		complex *w = spinor_init(length);
		complex *w0 = spinor_init(length);
		complex *x0 = spinor_init(length);
		complex *u0 = su3_init(length);
		spinor_conv(x0, x, XSLOW_TO_XFAST);
		su3_conv(u0, u, XSLOW_TO_XFAST);

		mul_su3_spinor_reor(w0, u0, x0);
		t = stop_watch(0);
		for(int i=0; i<niters; i++)
			mul_su3_spinor_reor(w0, u0, x0);
		t = stop_watch(t);

		spinor_conv(w, w0, XFAST_TO_XSLOW);
		d = diff(y,w);
  
		printf(" 4: nthr = %6d, %8lu iters, length = %8lu, t = %10.2lf usec per iter, BW = %6.2lf GB/s, perf = %12.2lf Mflop/s, diff = %e\n", nthr, niters, length, t/(double)niters, (2*NS*NC+NC*NC)*sizeof(complex)*length*niters/t*1e6/1024/1024/1024, 66*(double)niters*NS*(double)length/t, d);

		free(w);
		free(u0);
		free(w0);
		free(x0);    
	#endif

	#if defined(REOR) && (defined(SSE) || defined(MIC))
		w = spinor_init(length);
		w0 = spinor_init(length);
		x0 = spinor_init(length);
		u0 = su3_init(length);
	
		spinor_conv(x0, x, XSLOW_TO_XFAST);
		su3_conv(u0, u, XSLOW_TO_XFAST);

		mul_su3_spinor_reor_intrins(w0, u0, x0);
		t = stop_watch(0);
		for(int i=0; i<niters; i++)
			mul_su3_spinor_reor_intrins(w0, u0, x0);
		t = stop_watch(t);

		spinor_conv(w, w0, XFAST_TO_XSLOW);
		d = diff(y,w);
  
		printf(" 5: nthr = %6d, %8lu iters, length = %8lu, t = %10.2lf usec per iter, BW = %6.2lf GB/s, perf = %12.2lf Mflop/s, diff = %e\n", nthr, niters, length, t/(double)niters, (2*NS*NC+NC*NC)*sizeof(complex)*length*niters/t*1e6/1024/1024/1024, 66*(double)niters*NS*(double)length/t, d);

		free(w);
		free(u0);
		free(w0);
		free(x0);    
	#endif
	
	
	
	//data reordered using hopping
	#if defined(REOR) && defined(MIC)
	
		w = spinor_init(length);
		w0 = spinor_init(length);
		x0 = spinor_init(length);
		u0 = su3_init(length);

		spinor_conv(x0, x, HOPPING_XSLOW_TO_XFAST);
		su3_conv(u0, u, HOPPING_XSLOW_TO_XFAST);
	
		mul_su3_spinor_hopping_reor_intrins(w0, u0, x0);
		t = stop_watch(0);
		for(int i=0; i<niters; i++)
			mul_su3_spinor_hopping_reor_intrins(w0, u0, x0);
		t = stop_watch(t);
	
		spinor_conv(w, w0, HOPPING_XFAST_TO_XSLOW);
		d = diff(y,w);

		printf(" 6: nthr = %6d, %8lu iters, length = %8lu, t = %10.2lf usec per iter, BW = %6.2lf GB/s, perf = %12.2lf Mflop/s, diff = %e\n", nthr, niters, length, t/(double)niters, (2*NS*NC+NC*NC)*sizeof(complex)*length*niters/t*1e6/1024/1024/1024, 66*(double)niters*NS*(double)length/t, d);
		
		free(w);
		free(u0);
		free(w0);
		free(x0); 

	#endif
	
	free(x);
	free(y);
	free(u);
	

	return 0;
}
