#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <accel.h>

int main(int argc, char *argv[]) {
  int a = 1, m = 256, q = 256, n = 256, s = 1;
  float *A = NULL, *B = NULL, *C = NULL;
  struct timeval tv_comm_a, tv_comm_b, tv_mult_a, tv_mult_b;
  long long t_comm = 0, t_mult = 0;
  int num_devices, host_device, target_device;
  size_t p;

  host_device = -1;
  target_device = -1;
  num_devices = acc_get_num_devices(acc_device_nvidia);
  p = 1;

  if (num_devices > 0) {
    target_device = 0;
    acc_init(acc_device_nvidia);
    acc_set_device_num(target_device, acc_device_nvidia);
  }

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
    if (!strcmp("-s", argv[a])) {
      a++;
      if (a < argc) sscanf(argv[a], "%d", &s);
    }
    if (!strcmp("-p", argv[a])) {
      a++;
      if (a < argc) sscanf(argv[a], "%d", &p);
    }
    a++;
  }

  A = malloc(m * q * sizeof(float));
  B = malloc(q * n * sizeof(float));
  C = malloc(m * n * sizeof(float));
  memset(A, 0, m * q * sizeof(float));
  memset(B, 0, q * n * sizeof(float));
  memset(C, 0, m * n * sizeof(float));

  // Determine blocking based on number of OpenMP threads.
  // A and B are set up on b x c 2D grids.
  size_t b = floor(sqrt(m * p / n));
  size_t c = floor(sqrt(n * p / m));
  // Internal block dims. Each block of A is u x v cells. Each block of B is v x w cells.
  // NB: This will only work for square matrices.
  size_t u = m / b;
  size_t v = q / c;
  size_t w = n / c;
  size_t x,y;

  /* // Debug printout to check dimension values. */
  /* printf("p = %lu, b = %lu, c = %lu, u = %lu, v = %lu, w = %lu\n", p, b, c, u, v, w); */


  int i, j, k, l;
  // Fill A by block ID.
  for (i = 0; i < b; i++) {
    for (j = 0; j < c; j++) {
      for (k = 0; k < u; k++) {
  	for (l = 0; l < v; l++) {
  	  A[i * q * u + k * q + j * v + l] = i * c + j + 1;
  	}
      }
    }
  }

  // Fill B by block ID.
  for (i = 0; i < c; i++) {
    for (j = 0; j < b; j++) {
      for (k = 0; k < v; k++) {
  	for (l = 0; l < u; l++) {
  	  B[i * n * v + k * n + j * u + l] = i * c + j + 1;
  	}
      }
    }
  }

  /* for (i = 0; i < m; i++) { */
  /*   for (j = 0; j < q; j++) { */
  /*     A[i * q + j] = i * q + j + 1; */
  /*   } */
  /* } */

  /* for (i = 0; i < q; i++) { */
  /*   for (j = 0; j < n; j++) { */
  /*     B[i * n + j] = i * n + j + 1; */
  /*   } */
  /* } */

  // Phase 1: Offload the matrices to the target device and skew the matrices A and B.
  // If there is no target device, copy and skew to a location in host memory.
  float *restrict dA, *restrict dB, *restrict dC;

  // Allocate A with ghost column.
  dA = acc_malloc(m * (q+v) * sizeof(float));
  // Allocate B with ghost row.
  dB = acc_malloc((q+v) * n * sizeof(float));
  // Allocate C as-is.
  dC = acc_malloc(m * n * sizeof(float));

  fprintf(stdout, "p,m,q,n,trials,trial,flops,GF/s,t_mult_ms,t_comm_ms\n");

  int e;
  for (e = 0; e < s; e++) {

  gettimeofday(&tv_comm_a, NULL);

  // Copy C to target
  acc_memcpy_to_device(dC, C, m * n * sizeof(float));

  for (x = 0; x < b; x++) {
    for (y = 0; y < c; y++) {
      // Shear block A(x,y) to dA(x,(y-x)%c).
      // A remains in block row major, handled by P(x,y).

      for (k = 0; k < u; k++) {
	size_t dst_off = x * u * (q+v) + k * (q+v) + (abs((y - x) % c) + 1) * v;
	size_t src_off = x * u * q + k * q + y * v;
	acc_memcpy_to_device(dA + dst_off, A + src_off, v * sizeof(float));
      }

      for (k = 0; k < v; k++) {
	size_t dst_off = (abs((x - y) % b) + 1) * v * n + k * n + y * w;
	size_t src_off = x * v * n + k * n + y * w;
	acc_memcpy_to_device(dB + dst_off, B + src_off, w * sizeof(float));
      }
    }
  }
  gettimeofday(&tv_comm_b, NULL);

  t_comm += 1000000LL * (tv_comm_b.tv_sec - tv_comm_a.tv_sec) + tv_comm_b.tv_usec - tv_comm_a.tv_usec;
  /* // Debugging printout to validate shearing of A and B. */
  /* for (i = 0; i < m; i++) { */
  /*   printf("A = "); */
  /*   for (j = 0; j < q; j++) { */
  /*     printf("%f ", A[i * q + j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  /* for (i = 0; i < q; i++) { */
  /*   printf("B = "); */
  /*   for (j = 0; j < n; j++) { */
  /*     printf("%f ", B[i * n + j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  /* printf("===========================================\n"); */
  /* for (i = 0; i < m; i++) { */
  /*   printf("dA = "); */
  /*   for (j = 0; j < q + v; j++) { */
  /*     if (j == v) printf("| "); */
  /*     printf("%f ", dA[i * (q + v) + j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */

  /* for (i = 0; i < q + v; i++) { */
  /*   if (i == v) printf("-----------------------------------\n"); */
  /*   printf("dB = "); */
  /*   for (j = 0; j < n; j++) { */
  /*     printf("%f ", dB[i * n + j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */

  // Phase 2: Cycle through the blocks and multiply the blocks, shifting A left and B up by one block each iteration.
  size_t d;
  for (d = 0; d < sqrt(b*c); d++) {
    /* printf("===========================================\n"); */
    gettimeofday(&tv_comm_a, NULL);
    // Cycle A and B.
    for (y = 0; y <= c; y++) {
      for (x = 0; x < b; x++) {
	// Shift A(x,y) left 1 block.
	for (k = 0; k < u; k++) {
	  size_t dst_off = x * u * (q+v) + k * (q+v) + (y % (c+1)) * v;
	  size_t src_off = x * u * (q+v) + k * (q+v) + ((y + 1) % (c+1)) * v;
	  acc_memcpy_device(dA + dst_off, dA + src_off, v * sizeof(float));
	}
      }
    }

    for (x = 0; x <= b + 1; x++) {
      acc_memcpy_device(dB + (x%(b+1)) * v * n, dB + ((x+1)%(b+1)) * v * n, v * n * sizeof(float));
    }

    gettimeofday(&tv_comm_b, NULL);

    t_comm += 1000000LL * (tv_comm_b.tv_sec - tv_comm_a.tv_sec) + tv_comm_b.tv_usec - tv_comm_a.tv_usec;

    gettimeofday(&tv_mult_a, NULL);

    // Multiply all the blocks for the present iteration.
    // Block multiply A(x,y) by B(x,y).
#pragma acc kernels deviceptr(dC,dA,dB)
    for (i = 0; i < u; i++) {
#pragma acc loop independent
    for (x = 0; x < b; x++) {
#pragma acc loop independent
      for (y = 0; y < c; y++) {
	size_t coff = x * u * n + y * w + i * n;
	size_t aoff = x * u * (q+v) + (y+1) * v + i * (q+v);
	size_t boff = (x+1) * v * n + y * w;
#pragma acc loop independent
      for (k = 0; k < v; k++) {
#pragma acc loop independent
          for (j = 0; j < w; j++) {
	    *(dC + coff + j) += dA[aoff + k] * dB[boff + k * n + j];
	    }
          }

        }
      }
    }
    gettimeofday(&tv_mult_b, NULL);

    t_mult += 1000000LL * (tv_mult_b.tv_sec - tv_mult_a.tv_sec) + tv_mult_b.tv_usec - tv_mult_a.tv_usec;

    /* for (i = 0; i < m; i++) { */
    /*   printf("dA(%lu) = ", d); */
    /*   for (j = 0; j < q + v; j++) { */
    /* 	if (j == v) printf("| "); */
    /* 	printf(" %f ", dA[i * (q + v) + j]); */
    /*   } */
    /*   printf("\n"); */
    /* } */

    /* for (i = 0; i < q + v; i++) { */
    /*   if (i == v) printf("-----------------------------------\n"); */
    /*   printf("dB(%lu) = ", d); */
    /*   for (j = 0; j < n; j++) { */
    /* 	printf("%f ", dB[i * n + j]); */
    /*   } */
    /*   printf("\n"); */
    /* } */
  }

  gettimeofday(&tv_comm_a, NULL);

  // Copy results from device back to host.
  acc_memcpy_from_device(C, dC, m * n * sizeof(float));

  gettimeofday(&tv_comm_b, NULL);

  t_comm += 1000000LL * (tv_comm_b.tv_sec - tv_comm_a.tv_sec) + tv_comm_b.tv_usec - tv_comm_a.tv_usec;

  /* for (i = 0; i < m; i++) { */
  /*   printf("C = "); */
  /*   for (j = 0; j < n; j++) { */
  /*     printf("%f ", C[i * n + j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */

  // Report timing results.
  double flops = 2.0 * m * n * q;
  fprintf(stdout, "%lu,%d,%d,%d,%d,%d,%.2lf,%.2lf,%.3lf,%.3lf\n", p, m, q, n, s, e, flops, flops * 1e-9 / (t_mult * 1e-6), t_mult * 0.001, t_comm * 0.001);
  t_mult = 0;
  t_comm = 0;
  }

  if (dA) acc_free(dA);
  if (dB) acc_free(dB);
  if (dC) acc_free(dC);

  if (A) free(A);
  if (B) free(B);
  if (C) free(C);
  return 0;
}
