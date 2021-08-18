#Makefile for the interpolator script

CC=g++
CXXFLAGS=-O2 -Wall -Wextra 
LIBS=-larmadillo -llapack -lblas -lfftw3 -lconfig++ -lalglib
DEPS = InterpolatorClass.hpp InterpolatorSettings.hpp
OBJ = InterpolatorClass.o InterpolatorSettings.o

/%.o: %cpp $(DEPS)
	$(CC) -c $@ $< $(CFLAGS) 

main: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)



