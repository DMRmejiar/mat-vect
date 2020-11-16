CC = mpicc	
OPTS = -Wall

DEBUG ?= 1
ifeq ($(DEBUG), 1)
    OPTS += -g -O0
else
    OPTS += -O3
endif

all: my_it_mat_vect_mult_mpi.out my_it_mat_vect_mult.out

my_it_mat_vect_mult_mpi.out: my_it_mat_vect_mult_mpi.o
	$(CC) -o $@ $< $(LIBS)

my_it_mat_vect_mult.out: my_it_mat_vect_mult.o
	$(CC) -o $@ $< $(LIBS)

%.o: %.c
	$(CC) $(OPTS) -c $< -o $@ $(LIBS)

clean:
	rm -f *.o *.out *~

run:
	mpirun --oversubscribe -np 4 ./my_it_mat_vect_mult_mpi.out