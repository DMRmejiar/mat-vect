/**
 *   \file my_it_mat_vect_mult.c
 *   \brief Multiplica iterativamente un matriz nxn 
 *          por un vector de n posiciones
 *
 *   \author Danny Múnera
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>     /* For MPI functions, etc */ 

/* función para generar <size> cantidad de datos aleatorios */
void gen_data(double * array, int size);
/* función para multiplicar iterativamente un matriz 
 * <m x n> por un vector de tam <n> */
void mat_vect_mult(double* A, double* x, double* y, int n, int it, int my_rank, int comm_sz);
/* función para imprimir un vector llamado <name> de tamaño <m>*/
void print_vector(char* name, double*  y, int m);

int main()
{
  double* A = NULL;
  double* x = NULL;
  double* y = NULL;
  int n, iters;
  long seed;
  
  int comm_sz, my_rank;
  double local_start, local_finish, local_elapsed, elapsed;

  /* Start up MPI */
  MPI_Init(NULL, NULL);

  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // Obtener las dimensiones
  if (my_rank == 0)
  {
    printf("Ingrese la dimensión n:\n");
    scanf("%d", &n);
    printf("Ingrese el número de iteraciones:\n");
    scanf("%d", &iters);
    printf("Ingrese semilla para el generador de números aleatorios:\n");
    scanf("%ld", &seed);
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&seed, 1, MPI_LONG, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  srand(seed);

  // la matriz A tendrá una representación unidimensional
  A = malloc(sizeof(double) * n * n);
  x = malloc(sizeof(double) * n);
  y = malloc(sizeof(double) * n);

  //generar valores para las matrices
  gen_data(A, n*n);
  gen_data(x, n);
  
  local_start = MPI_Wtime();

  mat_vect_mult(A, x, y, n, iters, my_rank, comm_sz);
  
  local_finish = MPI_Wtime();
  
  local_elapsed = local_finish - local_start;
  
  MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (my_rank == 0)
  {
    print_vector("y", y, n);
    printf("Tiempo de ejecución = %5.2f segundos \n", elapsed);
  }
  free(A);
  free(x);
  free(y);

  /* Shut down MPI */
  MPI_Finalize();
  
  return 0;
}

void gen_data(double * array, int size){
  int i;
  for (i = 0; i < size; i++)
    array[i] = (double) rand() / (double) RAND_MAX;
}

void mat_vect_mult(double* A, double* x, double* y, int n, int it, int my_rank, int comm_sz)
{
  int h, i, j;
  double * local_y = NULL;
  int local_n, local_i, vect_size;

  local_y = malloc(sizeof(double) * n);

  for(i = 0; i < n; i++)
  {
    local_y[i] = 0.0;
  }

  vect_size = n/comm_sz;
  local_i = vect_size*my_rank;
  local_n = local_i + vect_size;
  
  if (my_rank == (comm_sz - 1))
    local_n += n%comm_sz;

  for(h = 0; h < it; h++)
  {
    for(i = local_i; i < local_n; i++)
    {
      local_y[i] = 0.0;
      for(j = 0; j < n; j++)
        local_y[i] += A[i*n+j] * x[j];
    }
    
    // x <= y
    for(i = 0; i < n; i++)
    {
      MPI_Allreduce(&local_y[i], &y[i], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      x[i] = y[i];
    }
  }
}

void print_vector(char* name, double*  y, int m) 
{
  int i;
  printf("\nVector %s\n", name);
  for (i = 0; i < m; i++)
    printf("%f ", y[i]);
  printf("\n");
}