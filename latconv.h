#ifndef _LATCONV_H
#define _LATCONV_H 1
#include <types.h>

void spinor_conv(complex *, complex *, enum conv);
void su3_conv(complex *, complex *, enum conv);
void su3_to_short_su3(complex *, complex *);
void copy4x3_to_4x4(complex* m4x3, complex* m4x4);
void copy3x3_to_3x4(complex* m3x3, complex* m3x4);
void copy4x4_to_4x3(complex* m4x4, complex* m4x3);
#endif /* _LATCONV_H */
