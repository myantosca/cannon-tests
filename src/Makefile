CC=gcc-7
MPICC=mpicc
ACC_INCS=-I/opt/pgi/linux-86-64/2017/include
ACC_LIBS=-L/opt/pgi/linux-86-64/2017/lib
all: cannon-su-acc cannon-so

cannon-su-omp: cannon-su.c
	@$(CC) -g -DOMP -fno-stack-protector -foffload=nvptx-none -fopenmp $^ -o $@ -lm -lgomp

cannon-su-acc: cannon-su.c
	@pgcc -g $(ACC_INCS) -DACC -ta=nvidia -mp -acc -Minfo=accel $^ -o $@ $(ACC_LIBS) -lm -lpgmp

cannon-so: cannon-so.c
	@$(MPICC) -g $^ -o $@ -lm -lmpi -lnuma

clean:
	@rm -f *~
	@rm -f cannon-su-acc
	@rm -f cannon-su-omp
	@rm -f cannon-so
