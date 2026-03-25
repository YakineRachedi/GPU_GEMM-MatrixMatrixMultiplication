CXXFLAGS = -std=c++14
ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -fbounds-check -pedantic -D_GLIBCXX_DEBUG 
CXXFLAGS2 = $(CXXFLAGS)
else
CXXFLAGS2 = ${CXXFLAGS} -O2 -march=native -Wall 
CXXFLAGS += -O3 -march=native -Wall
endif

ALL= test.exe

default:	help

all: $(ALL)

clean:
	@rm -rf *.o *.exe *~

test.exe: test.cpp
	@echo "Compiling executable..."
	$(CXX) $(CXXFLAGS2) $< -o $@


help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"