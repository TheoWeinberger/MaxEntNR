#Makefile for MaxEnt system 

CC=g++
CXXFLAGS=-O2 -Wall -Wextra -fopenmp
LIBS=-larmadillo -llapack -lblas -lfftw3 -lconfig++ -fopenmp
DEPS = MaxEntClass.hpp MaxEntSettings.hpp
OBJ = MaxEntMain.o MaxEntClass.o MaxEntSettings.o

/%.o: %cpp $(DEPS)
	$(CC) -c $@ $< $(CFLAGS) 

MaxEntMain: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)




