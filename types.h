#ifndef _TYPES_H
#define _TYPES_H 1

typedef struct {
  double re;
  double im;
} complex;

typedef struct {
  float re;
  float im;
} fcomplex;

enum conv {
  XFAST_TO_XSLOW,
  XSLOW_TO_XFAST,
  HOPPING_XFAST_TO_XSLOW,
  HOPPING_XSLOW_TO_XFAST
};

#endif /* _TYPES_H */
