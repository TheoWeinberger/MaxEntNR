CC=g++
CXXFLAGS=-O2 -Wall -Wextra -fopenmp
LIBS=-larmadillo -llapack -lblas -lfftw3 -lconfig++
DEPS = MaxEntClass.hpp MaxEntSettings.hpp
OBJ = MaxEntMain.o MaxEntClass.o MaxEntSettings.o

/%.o: %cpp $(DEPS)
	$(CC) -c $@ $< $(CXXFLAGS) 

MaxEntMain: $(OBJ)
	$(CC) -o $@ $^ $(CXXFLAGS) $(LIBS)




