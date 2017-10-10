#Source file
SRC+=code/src/main.cpp
SRC+=code/src/gbdt.cpp
SRC+=code/src/lr.cpp
SRC+=code/src/tree.cpp
SRC+=code/src/predictor.cpp
SRC+=code/src/utilis.cpp

#lib source
SRC+=liblinear/linear.cpp
SRC+=liblinear/tron.cpp
SRC+=liblinear/blas/daxpy.c
SRC+=liblinear/blas/ddot.c
SRC+=liblinear/blas/dnrm2.c
SRC+=liblinear/blas/dscal.c

#Object file
OBJ=$(SRC:.CPP=.O)

#Output execution file
PROGRAM=gbdt_predict

#Complier
CC=g++

#INCLUDE
OPENCV_HOME= #PATH
INCLUDE+=-Iliblinear -Icode/include
INCLUDE+=-I$(OPENCV_HOME)/include/opencv -I$(OPENCV_HOME)/include/opencv2

#CFLAGS
CFLAGS=-ansi -O -Wall `pkg-config --cflags opencv`

#LIBS(Linker Parameter)
LIBS+=-lpthread `pkg-config --libs opencv`

all:$(PROGRAM)
$(PROGRAM):$(OBJ)
	$(CC) $(INCLUDE) $(CFLAGS) $(OBJ) -o $(PROGRAM) $(LIBS)
.SUFFIXES:.cpp

clean:
	-rm *.o
