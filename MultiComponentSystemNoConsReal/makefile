CC=g++
CXXFLAGS=-O2 -Wall -Wextra -fopenmp
LIBS=-larmadillo -llapack -lblas -lfftw3 -lconfig++
DEPS = MaxEnt/MaxEntClass.hpp MaxEnt/MaxEntSettings.hpp Reflectivity/ReflectivityClass.hpp Reflectivity/ReflectivitySettings.hpp
OBJ = main.o MaxEnt/MaxEntClass.o MaxEnt/MaxEntSettings.o Reflectivity/ReflectivityClass.o Reflectivity/ReflectivitySettings.o

/%.o: %cpp $(DEPS)
	$(CC) $(CXXFLAGS) -c $< -o $(LDFLAGS) $@

main: $(OBJ)
	$(CC) -fopenmp -o $@ $^ $(CXXFLAGS) $(LIBS)


