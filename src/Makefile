CC=gcc-7
MPICC=mpicc
ACC_INCS=-I/opt/pgi/linux-86-64/2017/include

all: cannon-su-acc

cannon-su-omp: cannon-su.c
	@$(CC) -g -DOMP -fno-stack-protector -foffload=nvptx-none -fopenmp $^ -o $@ -lm -lgomp

cannon-su-acc: cannon-su.c
	pgcc -g $(ACC_INCS) -DACC $^ -o $@ -lm

cannon-so: cannon-so.c
	@$(MPICC) -g $^ -o $@ -lm -lmpi

clean:
	@rm -f cannon-su-acc
	@rm -f cannon-su-omp
	@rm -f cannon-so
