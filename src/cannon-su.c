#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <omp.h>

int main(int argc, char *argv[]) {
  int a = 1, m = 256, q = 256, n = 256, s = 30;
  float *A = NULL, *B = NULL, *C = NULL;

  int num_devices;
  int host_device = omp_get_initial_device();
  int target_device = host_device;
  num_devices = omp_get_num_devices();
  if (num_devices > 1) {
    target_device = 1;
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
    a++;
  }

  A = malloc(m * q * sizeof(float));
  B = malloc(q * n * sizeof(float));
  C = malloc(m * n * sizeof(float));
  memset(C, 0, m * n * sizeof(float));

  // Determine blocking based on number of OpenMP threads.
  // A is set up on a b x c 2D grid, B is set up on a c x b 2D grid.
  size_t p = omp_get_max_threads();
  //size_t b = m * p / n + (((m * p) % n) != 0);
  //size_t c = n * p / m + (((n * p) % m) != 0);
  size_t b = floor(sqrt(p));
  size_t c = p / b;
  // Internal block dims. Each block of A is u x v cells. Each block of B is v x u cells.
  size_t u = m / b;
  size_t v = q / c;
  size_t w = n / b;
  size_t x,y;

  // Debug printout to check dimension values.
  printf("p = %lu, b = %lu, c = %lu, u = %lu, v = %lu, w = %lu\n", p, b, c, u, v, w);


  // Fill A by block ID.
  int i, j, k, l;
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

  // Phase 1: Offload the matrices to the target device and skew the matrices A and B.
  // If there is no target device, copy and skew to a location in host memory.
  float *dA, *dB, *dC;

  // Allocate A with ghost column.
  dA = (float *)omp_target_alloc(m * (q+v) * sizeof(float), target_device);
  // Allocate B with ghost row.
  dB = (float *)omp_target_alloc(q * (n+w) * sizeof(float), target_device);
  // Allocate C as-is.
  dC = (float *)omp_target_alloc(m * n * sizeof(float), target_device);
  // Copy C to target
  omp_target_memcpy(dC, C, m * n * sizeof(float), 0, 0, target_device, host_device);
#pragma omp parallel for \
  default(none) num_threads(b)	   \
  shared(dA) shared(dB) shared(dC) \
  shared(A) shared(B) shared(C) \
  shared(host_device) shared(target_device) \
  shared(b) shared(c) shared(q) shared(u) shared(v) shared(w) \
  private(x)
  for (x = 0; x < b; x++) {
#pragma omp parallel for \
  default(none) num_threads(b)	   \
  shared(dA) shared(dB) shared(dC) \
  shared(A) shared(B) shared(C)	   \
  shared(host_device) shared(target_device) \
  shared(b) shared(c) shared(q) shared(u) shared(v) shared(w) \
  shared(x) private(y)
    for (y = 0; y < c; y++) {
      // Shear block A(x,y) to dA(x,(y+x)%c).
      // A remains in block row major, handled by P(x,y).
      omp_target_memcpy_rect(dA, A,                                                // dst, src
			     sizeof(float),                                        // elem size
			     2,                                                    // dims
			     (const size_t[2]){ u, v },                            // volume
			     (const size_t[2]){ x * u, (((y + x) % c) + 1) * v },  // dst offs
			     (const size_t[2]){ x * u, y * v },                    // src offs
			     (const size_t[2]){ b * u, (c + 1) * v},               // dst dims
			     (const size_t[2]){ b * u, c * v },                    // src dims
			     target_device,                                        // dst device
			     host_device);                                         // src device

      // Shear block B(y,x) to dB(x,(x+y)%c).
      // B swaps to block column major, handled by P(y,x).
      omp_target_memcpy_rect(dB, B,                                                // dst, src
			     sizeof(float),                                        // elems
			     2,                                                    // dims
			     (const size_t[2]){ v, w },                            // volume
			     (const size_t[2]){ x * v, (((x + y) % c) + 1) * w },  // dst offs
			     (const size_t[2]){ y * v, x * w },                    // src offs
			     (const size_t[2]){ b * v, (c + 1) * w },              // dst dims
			     (const size_t[2]){ c * v, b * w },                    // src dims
			     target_device,                                        // dst device
			     host_device);                                         // src device
    }
  }

  // Debugging printout to validate shearing of A and B.
  printf("A = \n");
  for (i = 0; i < m; i++) {
    for (j = 0; j < q; j++) {
      printf("%f ", A[i * q + j]);
    }
    printf("\n");
  }
  printf("dA = \n");
  for (i = 0; i < m; i++) {
    for (j = 0; j < q + v; j++) {
      printf("%f ", dA[i * (q + v) + j]);
    }
    printf("\n");
  }
  printf("B = \n");
  for (i = 0; i < q; i++) {
    for (j = 0; j < n; j++) {
      printf("%f ", B[i * n + j]);
    }
    printf("\n");
  }
  printf("dB = \n");
  for (i = 0; i < q; i++) {
    for (j = 0; j < n + w; j++) {
      printf("%f ", dB[i * (n + w) + j]);
    }
    printf("\n");
  }

  // Phase 2: Cycle through the blocks and multiply the blocks, shifting A left and B up (technically, left) by one block each iteration.
  size_t d;
  for (d = 0; d < c; d++) {
    // Multiply all the blocks for the present iteration.
    for (x = 0; x < b; x++) {
      for (y = 0; y < c; y++) {
	// Block multiply A(x,y) by B(x,y).
	for (i = 0; i < u; i++) {
	  for (k = 0; k < v; k++) {
	    for (j = 0; j < w; j++) {
	      dC[x * u * n + i * n + y * w + j] += dA[x * u * (q+v) + i * (q+v) + (y+1) * v + k] * dB[y * (n+w) * v + k * (n+w) + (x+1) * w + j]; 
	    }
	  }
	}
      }
    }
    // Cycle A and B.
    for (x = 0; x < b; x++) {
      for (y = 1; y <= c+1; y++) {
	// Shift A(x,y) to the left 1 block.
	omp_target_memcpy_rect(dA, dA,                                               // dst, src
			       sizeof(float),                                        // elem size
			       2,                                                    // dims
			       (const size_t[2]){ u, v },                            // volume
			       (const size_t[2]){ x * u, ((y - 1) % (c+1)) * v },    // dst offs
			       (const size_t[2]){ x * u, (y % (c+1)) * v },          // src offs
			       (const size_t[2]){ b * u, (c + 1) * v },              // dst dims
			       (const size_t[2]){ b * u, (c + 1) * v },              // src dims
			       target_device,                                        // dst device
			       target_device);                                       // src device
	// Shift B(x,y) up (to the left) 1 block.
	omp_target_memcpy_rect(dB, dB,                                               // dst, src
			       sizeof(float),                                        // elems
			       2,                                                    // dims
			       (const size_t[2]){ v, w },                            // volume
			       (const size_t[2]){ x * v, ((y - 1) % (c+1)) * w },    // dst offs
			       (const size_t[2]){ x * v, (y % (c+1)) * w },          // src offs
			       (const size_t[2]){ b * v, (c + 1) * w },              // dst dims
			       (const size_t[2]){ b * v, (c + 1) * w },              // src dims
			       target_device,                                        // dst device
			       target_device);                                       // src device
      }
    }
    printf("[%lu] dA = \n", d);
    for (i = 0; i < m; i++) {
      for (j = 0; j < q + v; j++) {
	printf("%f ", dA[i * (q + v) + j]);
      }
      printf("\n");
    }
    printf("[%lu] dB = \n", d);
    for (i = 0; i < q; i++) {
      for (j = 0; j < n + w; j++) {
	printf("%f ", dB[i * (n + w) + j]);
      }
      printf("\n");
    }
  }

  // Copy results from device back to host.
  omp_target_memcpy(C, dC, m * n * sizeof(float), 0, 0, host_device, target_device);

  printf("C = \n");
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      printf("%f ", C[i * n + j]);
    }
    printf("\n");
  }

  // Report timing results.
  // TODO

  if (A) free(A);
  if (B) free(B);
  if (C) free(C);
  if (dA) omp_target_free(dA, target_device);
  if (dB) omp_target_free(dB, target_device);
  if (dC) omp_target_free(dC, target_device);
  return 0;
}
