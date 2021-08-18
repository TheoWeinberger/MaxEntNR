#Makefile for DataMaskMain which links the DataMask class and Settings files

CC=g++
CXXFLAGS=-O2 -Wall -Wextra 
LIBS=-larmadillo -llapack -lblas -lfftw3 -lconfig++ -lalglib
DEPS = DataMask.hpp DataMaskSettings.hpp
OBJ = DataMask.o DataMaskSettings.o

/%.o: %cpp $(DEPS)
	$(CC) -c $@ $< $(CFLAGS) 

DataMaskMain: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)



