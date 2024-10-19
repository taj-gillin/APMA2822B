#include <stdio.h> 
#include <math.h>

int main(int argc, char **argv){
  int i,N = 10;
  double dx = 0.1;

  double x[10]; //request for statically allocating 10 elements of a type double

  double *y; //pointer to memory
  y = new double[10]; //request for dynamically allocating 10 elements of a type double

  for (i = 0; i < N; i++) x[i] = 1;
  for (i = 0; i < N; i++) y[i] = 2;

  // Dot product
  double sum = 0.0;
  for (i = 0; i < N; i++) sum += x[i]*y[i];

  printf("Dot product = %f\n", sum);

  delete[] y;//free dynamically allocated memory
  return 0;
}

