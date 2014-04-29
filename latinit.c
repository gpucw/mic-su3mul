#include <stdlib.h>
#include <utils.h>
#include <types.h>
#include <globals.h>

complex * spinor_init(size_t size)
{
	return alloc(size*NC*NS*sizeof(complex));
}

complex * su3_init(size_t size)
{
	return alloc(size*NC*NC*sizeof(complex));
}

complex * su3_short_init(size_t size)
{
	return alloc(size*(NC*NC-1)*sizeof(complex));
}
