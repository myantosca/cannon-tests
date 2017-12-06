MKLROOT=/opt/intel/mkl
MKLLIBS=-L$(MKLROOT)/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

all: mkl_cblas_sgemm

mkl_cblas_sgemm: mkl_cblas_sgemm.c

	gcc -g -I$(MKLROOT)/include $^ -o mkl_cblas_sgemm $(MKLLIBS)

clean:
	@rm -f mkl_cblas_sgemm
