#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define ALIGNMENT 512

/*
 * Allocate an aligned chunk and check if NULL returned
 */
void * alloc(size_t size)
{
  void *ptr;
  posix_memalign(&ptr, ALIGNMENT, size);

  if(ptr == NULL) {
    fprintf(stderr, " alloc() returned NULL. Out of memory?\n");
  }
   
  return ptr;
}

/*
 * Returns the time in micro seconds since the argument "t"
 */
double stop_watch(double t) 
{
  struct timeval tv;
  gettimeofday(&tv, NULL); 
  return (tv.tv_sec*1000*1000 + tv.tv_usec) - t;
}
