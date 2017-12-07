#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  int a = 1, m = 256, q = 256, n = 256, s = 1;
  float *A = NULL, *B = NULL, *C = NULL;
  //struct timeval tv_comm_a, tv_comm_b, tv_mult_a, tv_mult_b;
  double t_comm = 0.0, t_mult = 0.0, t_comm_a, t_comm_b, t_comm_a_min, t_comm_b_max, t_mult_a, t_mult_b, t_mult_a_min, t_mult_b_max;
  int p;
  int world_rank, world_size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  t_comm_a = MPI_Wtime();
  t_mult_a = MPI_Wtime();


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

  // Determine blocking based on number of MPI processes.
  int pdims[2]={0,0};
  MPI_Dims_create(p, 2, pdims);
  size_t b = pdims[0];
  size_t c = pdims[1] ? pdims[1] : 1;

  // A and B are set up on b x c 2D grids.
  // Internal block dims. Each block of A is u x v cells. Each block of B is v x w cells.
  // NB: This will only work for square matrices.
  size_t u = m / b;
  size_t v = q / c;
  size_t w = n / c;
  size_t x,y,px,py;
  int i, j, k, l;

  px = world_rank / b;
  py = world_rank % b;
  /* // Debug printout to check dimension values. */
  /* printf("p = %lu, b = %lu, c = %lu, u = %lu, v = %lu, w = %lu\n", p, b, c, u, v, w); */

  // Only the root process fills in the initial matrices.
  // Each chunk will be delivered to its respective process.
  if (world_rank == 0) {
    A = malloc(m * q * sizeof(float));
    B = malloc(q * n * sizeof(float));
    C = malloc(m * n * sizeof(float));
    memset(A, 0, m * q * sizeof(float));
    memset(B, 0, q * n * sizeof(float));
    memset(C, 0, m * n * sizeof(float));

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

  }



  // Phase 1: Offload chunks of the matrices A and B to the various processes,
  // skewing them with respect to the process grid.
  float *dA, *dB, *dC;
  dA = (float *)malloc((u*2) * v * sizeof(float));
  dB = (float *)malloc((v*2) * w * sizeof(float));
  dC = (float *)malloc((u*2) * w * sizeof(float));

  MPI_Group world_group, root_group, asrc_group, bsrc_group, adst_group, bdst_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  MPI_Group_incl(world_group, 1, (int[1]){ 0 }, &root_group);
  MPI_Group_incl(world_group, 1, (int[1]){ px * c + ((py+1)%c) }, &asrc_group);
  MPI_Group_incl(world_group, 1, (int[1]){ ((px+1)%b) * c + py }, &bsrc_group);
  MPI_Group_incl(world_group, 1, (int[1]){ px * c + ((py-1+c)%c) }, &adst_group);
  MPI_Group_incl(world_group, 1, (int[1]){ ((px-1+b)%b) * c + py }, &bdst_group);

  //printf("(%lu,%lu) <= (%lu,%lu) <= (%lu,%lu)\n", px, ((py-1+c)%c), px, py, px, ((py+1)%c));

  MPI_Win win_dA, win_dB, win_C, win_A, win_B;
  MPI_Win_create(dA, u * v * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win_dA);
  MPI_Win_create(dB, v * w * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win_dB);
  MPI_Win_create(C, world_rank != 0 ? 0 : m * n * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win_C);
  MPI_Win_create(A, world_rank != 0 ? 0 : m * q * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win_A);
  MPI_Win_create(B, world_rank != 0 ? 0 : q * n * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win_B);
  int e;

  for (e = 0; e < s; e++) {

    memset(dA, 0x0, (u*2) * v * sizeof(float));
    memset(dB, 0x0, (v*2) * w * sizeof(float));
    memset(dC, 0x0, (u*2) * w * sizeof(float));

    t_comm_a = MPI_Wtime();

    if (world_rank == 0) {
      MPI_Win_post(world_group, 0, win_A);
      MPI_Win_post(world_group, 0, win_B);
      MPI_Win_post(world_group, 0, win_C);
    }

    MPI_Win_start(root_group, 0, win_A);
    MPI_Win_start(root_group, 0, win_B);
    MPI_Win_start(root_group, 0, win_C);

    // Retrieve block C(px,py).
    for (k = 0; k < u; k++) {
      size_t off = (px * u * n + k * n + py * w);
      MPI_Get(dC + k * w, w, MPI_FLOAT, 0, off, w, MPI_FLOAT, win_C);
    }

    // Retrieve block A(px,(py-px)%c).
    // A remains in block row major, handled by P(x,y).
    for (k = 0; k < u; k++) {
      size_t shear_off = (px * u * q + k * q + ((py+px)%c) * v);
      MPI_Get(dA + k * v, v, MPI_FLOAT, 0, shear_off, v, MPI_FLOAT, win_A);
    }

    // Retrieve block B((px-py)%b,py).
    for (k = 0; k < v; k++) {
      size_t shear_off = (((px+py)%b) * v * n + k * n + py * w);
      MPI_Get(dB + k * w, w, MPI_FLOAT, 0, shear_off, w, MPI_FLOAT, win_B);
    }

    MPI_Win_complete(win_A);
    MPI_Win_complete(win_B);
    MPI_Win_complete(win_C);

    if (world_rank == 0) {
      MPI_Win_wait(win_A);
      MPI_Win_wait(win_B);
      MPI_Win_wait(win_C);
    }

    t_comm_b = MPI_Wtime();

    MPI_Reduce(&t_comm_a, &t_comm_a_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm_b, &t_comm_b_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
      t_comm += t_comm_b_max - t_comm_a_min;
    }
    /* // Debugging printout to validate shearing of A and B. */
    /* for (i = 0; i < u; i++) { */
    /*   printf("dA(%lu,%lu) = ", px, py); */
    /*   for (j = 0; j < v; j++) { */
    /*     printf("%f ", dA[i * v + j]); */
    /*   } */
    /*   printf("\n"); */
    /* } */

    /* for (i = 0; i < v; i++) { */
    /*   printf("dB(%lu,%lu) = ", px, py); */
    /*   for (j = 0; j < w; j++) { */
    /*     printf("%f ", dB[i * w + j]); */
    /*   } */
    /*   printf("\n"); */
    /* } */

    /* printf("(%lu,%lu) Phase 2\n", px, py); */
    // Phase 2: Cycle through the blocks and multiply the blocks, shifting A left and B up by one block each iteration.
    size_t d;
    for (d = 0; d < sqrt(b*c); d++) {
      t_comm_a = MPI_Wtime();

      memcpy(dA + u * v, dA, u * v * sizeof(float));
      MPI_Win_post(asrc_group, 0, win_dA);
      memcpy(dB + v * w, dB, v * w * sizeof(float));
      MPI_Win_post(bsrc_group, 0, win_dB);

      // Cycle A and B.
      // Shift A(x,y) left 1 block.
      MPI_Win_start(adst_group, 0, win_dA);
      MPI_Put(dA + u * v, u * v, MPI_FLOAT, px * c + ((py-1+c)%c), 0, u * v, MPI_FLOAT, win_dA);
      MPI_Win_complete(win_dA);

      // Shift B(x,y) up 1 block.
      MPI_Win_start(bdst_group, 0, win_dB);
      MPI_Put(dB + v * w, v * w, MPI_FLOAT, ((px-1+b)%b) * c + py, 0, v * w, MPI_FLOAT, win_dB);
      MPI_Win_complete(win_dB);

      MPI_Win_wait(win_dA);
      MPI_Win_wait(win_dB);

      t_comm_b = MPI_Wtime();

      MPI_Reduce(&t_comm_a, &t_comm_a_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
      MPI_Reduce(&t_comm_b, &t_comm_b_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      if (world_rank == 0) {
	t_comm += t_comm_b_max - t_comm_a_min;
      }

      t_mult_a = MPI_Wtime();

      // Block multiply A(x,y) by B(x,y).
      for (i = 0; i < u; i++) {
	for (k = 0; k < v; k++) {
	  for (j = 0; j < w; j++) {
	    dC[i * w + j] += dA[i * v + k] * dB[k * w + j];
	  }
	}
      }

      t_mult_b = MPI_Wtime();

      MPI_Reduce(&t_mult_a, &t_mult_a_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
      MPI_Reduce(&t_mult_b, &t_mult_b_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      if (world_rank == 0) {
	t_mult += t_mult_b_max - t_mult_a_min;
      }
      /* for (i = 0; i < u; i++) { */
      /*   printf("dA(%lu,%lu) = ", px, py); */
      /*   for (j = 0; j < v; j++) { */
      /* 	  printf(" %f ", dA[i * v + j]); */
      /*   } */
      /*   printf("\n"); */
      /* } */

      /* for (i = 0; i < v; i++) { */
      /*   printf("dB(%lu,%lu) = ", px, py); */
      /*   for (j = 0; j < w; j++) { */
      /* 	  printf("%f ", dB[i * w + j]); */
      /*   } */
      /*   printf("\n"); */
      /* } */

      /* for (i = 0; i < u; i++) { */
      /*   printf("dC(%lu,%lu) = ", px, py); */
      /*   for (j = 0; j < w; j++) { */
      /* 	  printf("%f ", dC[i * w + j]); */
      /*   } */
      /*   printf("\n"); */
      /* } */
    }

    // Copy results from device back to host.
    // Send block dC(px,py) to C(px,py).
    t_comm_a = MPI_Wtime();

    if (world_rank == 0) {
      MPI_Win_post(world_group, 0, win_C);
    }

    MPI_Win_start(root_group, 0, win_C);
    for (k = 0; k < u; k++) {
      size_t off = (px * u * n + k * n + py * w);
      MPI_Put(dC + k * w, w, MPI_FLOAT, 0, off, w, MPI_FLOAT, win_C);
    }
    MPI_Win_complete(win_C);

    if (world_rank == 0) {
      MPI_Win_wait(win_C);
    }

    t_comm_b = MPI_Wtime();

    MPI_Reduce(&t_comm_a, &t_comm_a_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm_b, &t_comm_b_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
      t_comm += t_comm_b_max - t_comm_a_min;
    }

    /* if (world_rank == 0) { */
    /*   for (i = 0; i < m; i++) { */
    /* 	printf("C = "); */
    /* 	for (j = 0; j < n; j++) { */
    /* 	  printf("%f ", C[i * n + j]); */
    /* 	} */
    /* 	printf("\n"); */
    /*   } */
    /* } */

    // Report timing results.
    double flops = 2.0 * m * n * q;
    if (world_rank == 0) {
      fprintf(stdout, "%d,%lu,%lu,%d,%d,%d,%d,%d,%.2lf,%.2lf,%.3lf,%.3lf\n", p, px, py, world_rank, m, q, n, s, flops, flops * 1e-9 / t_mult, t_mult * 1000, t_comm * 1000);
    }
  }

  if (A) free(A);
  if (B) free(B);
  if (C) free(C);
  if (dA) free(dA);
  if (dB) free(dB);
  if (dC) free(dC);

  MPI_Finalize();
  return 0;
}
