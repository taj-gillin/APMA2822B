#include <stdio.h> 
#include <math.h>
#include <sys/time.h>
#include <omp.h>


int main(int argc, char **argv){
  int i,N = 10;
  double dx = 0.1;
  
  size_t N1 = (size_t) 1024*1024*1024/sizeof(double);

  double *x; //dynamic memory
  double *y; //pointer to memory

  struct timeval t1, t2;

  x = new double[N1]; //request for dynamically allocating N1 elements of a type double
  gettimeofday(&t1, NULL);
  y = new double[N1]; //request for dynamically allocating N1 elements of a type double

  gettimeofday(&t2, NULL);

  fprintf(stderr,"ellapsed time : %ld [microseconds] \n", ((t2.tv_sec * 1000000 + t2.tv_usec)
		  - (t1.tv_sec * 1000000 + t1.tv_usec)));


  // for (i = 0; i < N; i++) x[i] = dx*i;
  // for (i = 0; i < N; i++) y[i] = sin( x[i] );

  delete[] y;//free dynamically allocated memory
  return 0;
}

