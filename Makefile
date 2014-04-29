### Intel Xeon Phi
CC=icc -DOMP -mmic -std=gnu99 -DREOR -DMIC
CFLAGS=-O3 -I./ -openmp -opt-prefetch-distance=64,8 -opt-streaming-cache-evict=0 -opt-streaming-stores always
LIBS=-openmp
LDFLAGS= 

### Intel CPU, GCC
#CC=gcc -std=gnu99 -DOMP -DSSE -DREOR
#CFLAGS=-O3 -I./ -fopenmp -msse3
#LIBS=-fopenmp -lm
#LDFLAGS= 

SOURCES=main \
	utils \
	latinit \
	latrand \
	latconv \
	mul_su3_spinor

main: ${addsuffix .o, $(SOURCES)}
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	$(RM) -f *.o main
