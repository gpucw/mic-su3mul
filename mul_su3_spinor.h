#ifndef _MUL_SU3_SPINOR_H
#define _MUL_SU3_SPINOR_H 1
#include <types.h>

void mul_su3_spinor(complex *, complex *, complex *);
void mul_short_su3_spinor(complex *, complex *, complex *);
void mul_su3_spinor_reor(complex *, complex *, complex *);
void mul_su3_spinor_intrins(complex *, complex *, complex *);
void mul_short_su3_spinor_intrins(complex *, complex *, complex *);
void mul_su3_spinor_reor_intrins(complex *, complex *, complex *);
void mul_su3_spinor_hopping_reor_intrins(complex *y, complex *u, complex *x);

#endif /* _MUL_SU3_SPINOR_H */
