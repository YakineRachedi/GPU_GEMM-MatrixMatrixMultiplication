CXX      = g++
NVCC     = nvcc

CXXFLAGS = -std=c++14
ifdef DEBUG
CXXFLAGS += -g -O0 -Wall -fbounds-check -pedantic -D_GLIBCXX_DEBUG -DSCALAR_TYPE=$(TYPE)
CXXFLAGS2 = $(CXXFLAGS)
else
CXXFLAGS2 = ${CXXFLAGS} -O2 -march=native -Wall 
CXXFLAGS += -O3 -march=native -Wall
endif

ARCH         ?= sm_80 
NVCCFLAGS     = -std=c++14 -arch=$(ARCH) -O2
ifdef DEBUG
NVCCFLAGS    += -g -G 
endif

ALL= test.exe cuda_test.exe

default:	help

all: $(ALL)

clean:
	@rm -rf *.o *.exe *~

test.exe: test.cpp
	@echo "Compiling CPU executable..."
	$(CXX) $(CXXFLAGS2) $< -o $@

cuda_test: cuda_test.cu
	@echo "Compiling CUDA executable..."
	$(NVCC) $(NVCCFLAGS) $< -o $@

help:
	@echo "Available targets :"
	@echo "    all          : compile CPU + CUDA"
	@echo "    test.exe     : CPU only"
	@echo "    cuda_test    : CUDA only"
	@echo ""
	@echo "Options :"
	@echo "    DEBUG=yes        : mode debug"
	@echo "    TYPE=float|double: type scalar (default: float)"
	@echo "    ARCH=sm_XX       : architecture GPU (default: sm_80)"