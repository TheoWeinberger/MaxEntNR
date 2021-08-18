#Makefile for the reflectivity class

CC=g++
CXXFLAGS=-O2 -Wall -Wextra 
LIBS=-larmadillo -llapack -lblas -lfftw3 -lconfig++
DEPS = ReflectivityClass.hpp ReflectivitySettings.hpp
OBJ = ReflectivityMain.o ReflectivityClass.o ReflectivitySettings.o

/%.o: %cpp $(DEPS)
	$(CC) -c $@ $< $(CFLAGS) 

ReflectivityMain: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)
