CC=gcc-7
MPICC=mpicc
ACC_INCS=-I/opt/pgi/linux-86-64/2017/include
ACC_LIBS=-L/opt/pgi/linux-86-64/2017/lib
all: cannon-su cannon-so

cannon-su: cannon-su.c
	@pgcc -g $(ACC_INCS) -ta=tesla,cuda8.0 -acc -Minfo=accel $^ -o $@ $(ACC_LIBS) -lm

cannon-so: cannon-so.c
	@$(MPICC) -g $^ -o $@ -lm -lmpi -lnuma

clean:
	@rm -f *~
	@rm -f *.o
	@rm -f cannon-su
	@rm -f cannon-so
