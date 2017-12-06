#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#ifdef OMP
#include <omp.h>
#endif

#ifdef ACC
#include <accel.h>
#endif

int main(int argc, char *argv[]) {
  int a = 1, m = 256, q = 256, n = 256, s = 1;
  float *A = NULL, *B = NULL, *C = NULL;
  struct timeval tv_comm_a, tv_comm_b, tv_mult_a, tv_mult_b;
  long long t_comm = 0, t_mult = 0;
  int num_devices, host_device, target_device;
  size_t p;

#ifdef OMP
  host_device = omp_get_initial_device();
  target_device = host_device;
  num_devices = omp_get_num_devices();
  p = omp_get_max_threads();
#endif

#ifdef ACC
  host_device = -1;
  target_device = -1;
  num_devices = acc_get_num_devices(acc_device_nvidia);
  p = 1;
#endif

  if (num_devices > 0) {
    target_device = 0;
#ifdef ACC
    acc_init(acc_device_nvidia);
    acc_set_device_num(target_device, acc_device_nvidia);
#endif
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
#ifdef ACC
    if (!strcmp("-p", argv[a])) {
      a++;
      if (a < argc) sscanf(argv[a], "%d", &p);
    }
#endif
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

  int e;
  for (e = 0; e < s; e++) {
  // Phase 1: Offload the matrices to the target device and skew the matrices A and B.
  // If there is no target device, copy and skew to a location in host memory.
  float *dA, *dB, *dC;

  gettimeofday(&tv_comm_a, NULL);
#ifdef OMP
  // Allocate A with ghost column.
  dA = (float *)omp_target_alloc(m * (q+v) * sizeof(float), target_device);
  // Allocate B with ghost row.
  dB = (float *)omp_target_alloc((q+v) * n * sizeof(float), target_device);
  // Allocate C as-is.
  dC = (float *)omp_target_alloc(m * n * sizeof(float), target_device);
  // Copy C to target
  omp_target_memcpy(dC, C, m * n * sizeof(float), 0, 0, target_device, host_device);
#endif

#ifdef ACC
  // Allocate A with ghost column.
  dA = acc_malloc(m * (q+v) * sizeof(float));
  // Allocate B with ghost row.
  dB = acc_malloc((q+v) * n * sizeof(float));
  // Allocate C as-is.
  dC = acc_malloc(m * n * sizeof(float));
  // Copy C to target
  acc_memcpy_to_device(dC, C, m * n * sizeof(float));
#endif

#ifdef OMP
#pragma omp parallel for	   \
  default(none) num_threads(b)	   \
  shared(dA) shared(dB) shared(dC) \
  shared(A) shared(B) shared(C) \
  shared(host_device) shared(target_device) \
  shared(b) shared(c) shared(q) shared(u) shared(v) shared(w) \
  private(x)
#endif
  for (x = 0; x < b; x++) {
#ifdef OMP
#pragma omp parallel for	   \
  default(none) num_threads(b)	   \
  shared(dA) shared(dB) shared(dC) \
  shared(A) shared(B) shared(C)	   \
  shared(host_device) shared(target_device) \
  shared(b) shared(c) shared(q) shared(u) shared(v) shared(w) \
  shared(x) private(y)
#endif
    for (y = 0; y < c; y++) {
      // Shear block A(x,y) to dA(x,(y-x)%c).
      // A remains in block row major, handled by P(x,y).
#ifdef OMP
      omp_target_memcpy_rect(dA, A,                                                // dst, src
			     sizeof(float),                                        // elem size
			     2,                                                    // dims
			     (const size_t[2]){ u, v },                            // volume
			     (const size_t[2]){ x * u, (abs((y - x) % c) + 1) * v },  // dst offs
			     (const size_t[2]){ x * u, y * v },                    // src offs
			     (const size_t[2]){ b * u, (c + 1) * v},               // dst dims
			     (const size_t[2]){ b * u, c * v },                    // src dims
			     target_device,                                        // dst device
			     host_device);                                         // src device

      // Shear block B(y,x) to dB((x-y)%b,x).
      // B remains in block row major, handled by P(x,y).
      omp_target_memcpy_rect(dB, B,                                                // dst, src
			     sizeof(float),                                        // elems
			     2,                                                    // dims
			     (const size_t[2]){ v, w },                            // volume
			     (const size_t[2]){ (abs((x - y) % b) + 1) * v, y * w },  // dst offs
			     (const size_t[2]){ x * v, y * w },                    // src offs
			     (const size_t[2]){ (b + 1) * v, c * w },              // dst dims
			     (const size_t[2]){ b * v, c * w },                    // src dims
			     target_device,                                        // dst device
			     host_device);                                         // src device
#endif

#ifdef ACC
      // TODO: Do phase 1 shear with OpenACC.
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
#endif
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
      #pragma acc kernels num_gangs(b) num_workers(u)
      for (x = 0; x < b; x++) {
	// Shift A(x,y) left 1 block.
#ifdef OMP
	omp_target_memcpy_rect(dA, dA,                                               // dst, src
			       sizeof(float),                                        // elem size
			       2,                                                    // dims
			       (const size_t[2]){ u, v },                            // volume
			       (const size_t[2]){ x * u, (y % (c+1)) * v },          // dst offs
			       (const size_t[2]){ x * u, ((y + 1) % (c+1)) * v },    // src offs
			       (const size_t[2]){ b * u, (c + 1) * v },              // dst dims
			       (const size_t[2]){ b * u, (c + 1) * v },              // src dims
			       target_device,                                        // dst device
			       target_device);                                       // src device
#endif

#ifdef ACC
	for (k = 0; k < u; k++) {
	  size_t dst_off = x * u * (q+v) + k * (q+v) + (y % (c+1)) * v;
	  size_t src_off = x * u * (q+v) + k * (q+v) + ((y + 1) % (c+1)) * v;
	  acc_memcpy_device(dA + dst_off, dA + src_off, v * sizeof(float));
	}
#endif
      }
    }


    for (x = 0; x <= b + 1; x++) {
#ifdef OMP
    // Shift B(x) up 1 block.
    omp_target_memcpy_rect(dB, dB,                                               // dst, src
                           sizeof(float),                                        // elems
                           2,                                                    // dims
                           (const size_t[2]){ v, c * w },                        // volume
			   (const size_t[2]){ (x % (b+1)) * v, y * w },          // dst offs
			   (const size_t[2]){ ((x + 1) % (b+1)) * v, y * w },    // src offs
			   (const size_t[2]){ (b + 1) * v, c * w },              // dst dims
			   (const size_t[2]){ (b + 1) * v, c * w },              // src dims
			   target_device,                                        // dst device
			   target_device);                                       // src device
#endif

#ifdef ACC
      acc_memcpy_device(dB + (x%(b+1)) * v * n, dB + ((x+1)%(b+1)) * v * n, v * n * sizeof(float));
#endif
    }

    gettimeofday(&tv_comm_b, NULL);

    t_comm += 1000000LL * (tv_comm_b.tv_sec - tv_comm_a.tv_sec) + tv_comm_b.tv_usec - tv_comm_a.tv_usec;

    gettimeofday(&tv_mult_a, NULL);

#ifdef OMP
    #pragma omp target parallel for device(target_device) num_threads(b) private(x) is_device_ptr(dC,dA,dB)
#endif
#ifdef ACC
    #pragma acc kernels num_gangs(b) num_workers(c) deviceptr(dC,dA,dB)
#endif
    // Multiply all the blocks for the present iteration.
    for (x = 0; x < b; x++) {
#ifdef OMP
      #pragma omp parallel for num_threads(c) private(y) private(i) private(k) private(j) firstprivate(x)
#endif
      for (y = 0; y < c; y++) {
	// Block multiply A(x,y) by B(x,y).
	for (i = 0; i < u; i++) {
	  for (k = 0; k < v; k++) {
	    for (j = 0; j < w; j++) {
	      //printf("[%lu] %f + ", x * u * n + i * n + y * w + j, dC[x * u * n + i * n + y * w + j]);
	      dC[x * u * n + i * n + y * w + j] += dA[x * u * (q+v) + i * (q+v) + (y+1) * v + k] * dB[(x+1) * v * n + k * n + y * w + j];
	      //printf("%f * %f = %f\n", dA[x * u * (q+v) + i * (q+v) + (y+1) * v + k], dB[(x+1) * v * n + k * n + y * w + j], dC[x * u * n + i * n + y * w + j]);
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

  // Copy results from device back to host.
#ifdef OMP
  omp_target_memcpy(C, dC, m * n * sizeof(float), 0, 0, host_device, target_device);
#endif
#ifdef ACC
  acc_memcpy_from_device(C, dC, m * n * sizeof(float));
#endif

  /* for (i = 0; i < m; i++) { */
  /*   printf("C = "); */
  /*   for (j = 0; j < n; j++) { */
  /*     printf("%f ", C[i * n + j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */

  // Report timing results.
  double flops = 2.0 * m * n * q;
  fprintf(stdout, "%lu,%d,%d,%d,%d,%.2lf,%.2lf,%.3lf,%.3lf\n", p, m, q, n, s, flops, flops * 1e-9 / (t_mult * 1e-6), t_mult * 0.001, t_comm * 0.001);
  }
  if (A) free(A);
  if (B) free(B);
  if (C) free(C);
#ifdef OMP
  if (dA) omp_target_free(dA, target_device);
  if (dB) omp_target_free(dB, target_device);
  if (dC) omp_target_free(dC, target_device);
#endif
  return 0;
}
