MKLROOT=/opt/intel/mkl
MKLLIBS=-L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

all: mkl_cblas_sgemm

mkl_cblas_sgemm: mkl_cblas_sgemm.c

	gcc -g -I$(MKLROOT)/include $^ -o mkl_cblas_sgemm $(MKLLIBS)
