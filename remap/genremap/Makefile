# Makefile for genremap program

# For GCC with aggressive optimization, uncomment these lines
#CXX = g++
#OPTFLAGS = -O3

# For Intel compiler with aggressive optimization, uncomment these lines
#CXX = icpc
#OPTFLAGS = -O3

CPPFLAGS =
CXXFLAGS = -Wall $(OPTFLAGS)
LDFLAGS =

all: genremap

genremap: genremap.o
	$(CXX) -o genremap $^ $(LDFLAGS)

clean:
	rm -f genremap *.o
