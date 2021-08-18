#Makefile for chisquared fitter

CC=g++
CXXFLAGS=-O2 -Wall -Wextra 
LIBS=-larmadillo -llapack -lblas -lfftw3 -lconfig++
DEPS = NewtonClass.hpp NewtonSettings.hpp
OBJ = NewtonMain.o NewtonClass.o NewtonSettings.o 

/%.o: %cpp $(DEPS)
	$(CC) -c $@ $< $(CFLAGS) 

NewtonMain: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)


