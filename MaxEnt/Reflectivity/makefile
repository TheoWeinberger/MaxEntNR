CC=g++
CXXFLAGS=-O2 -Wall -Wextra -fopenmp
LIBS=-larmadillo -llapack -lblas -lfftw3 -lconfig++
DEPS = ReflectivityClass.hpp ReflectivitySettings.hpp
OBJ = ReflectivityMain.o ReflectivityClass.o ReflectivitySettings.o

/%.o: %cpp $(DEPS)
	$(CC) -c $@ $< $(CXXFLAGS) 

ReflectivityMain: $(OBJ)
	$(CC) -o $@ $^ $(CXXFLAGS) $(LIBS)
