CC=gcc-7
MPICC=mpicc

all: cannon-su cannon-so

cannon-su: cannon-su.c
	@$(CC) -g -fno-stack-protector -foffload=nvptx-none -fopenmp $^ -o $@ -lm -lgomp

cannon-so: cannon-so.c
	@$(MPICC) -g $^ -o $@ -lm -lmpi
