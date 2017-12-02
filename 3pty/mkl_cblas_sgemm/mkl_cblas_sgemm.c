#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include <mkl.h>

int main(int argc, char *argv[]) {
  int a = 1, m = 256, q = 256, n = 256;
  float *A = NULL, *B = NULL, *C = NULL;

  while (a < argc) {
    if (!strcmp("-m", argv[a])) {
      a++;
      if (a < argc) sscanf(argv[a], "%d", &m);
    }
    if (!strcmp("-q", argv[a])) {
      a++;
      if (a < argc) sscanf(argv[a], "%d", &q);
    }
    if (!strcmp("-n", argv[a])) {
      a++;
      if (a < argc) sscanf(argv[a], "%d", &n);
    }
    a++;
  }

  A = malloc(m * q * sizeof(float));
  B = malloc(q * n * sizeof(float));
  C = malloc(m * n * sizeof(float));
  memset(C, 0, m * n * sizeof(float));

  int i, j;
  for (i = 0; i < m; i++) {
    for (j = 0; j < q; j++) {
      A[i * q + j] = 1.0;
    }
  }
  for (i = 0; i < q; i++) {
    for (j = 0; j < n; j++) {
      B[i * n + j] = 1.0;
    }
  }

  struct timeval tv_a;
  struct timeval tv_b;

  gettimeofday(&tv_a, NULL);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, q, 1, A, q, B, n, 0, C, n);
  gettimeofday(&tv_b, NULL);

  long long int t_us = 1000000LL * (tv_b.tv_sec - tv_a.tv_sec) + tv_b.tv_usec - tv_a.tv_usec;

  fprintf(stderr, "%d,%d,%d,%lld\n", m, q, n, t_us);

  if (A) free(A);
  if (B) free(B);
  if (C) free(C);

  return 0;
}
