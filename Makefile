CXX      = g++
NVCC     = nvcc

OPENBLAS_INC   = C:/OpenBLAS/include
OPENBLAS_LIB   = C:/OpenBLAS/lib
OPENBLAS_FLAGS = -lopenblas -static

TYPE     ?= float

CXXFLAGS = -std=c++14 -DSCALAR_TYPE=$(TYPE)

ifdef DEBUG
CXXFLAGS  += -g -O0 -Wall -fbounds-check -pedantic -D_GLIBCXX_DEBUG
CXXFLAGS2  = $(CXXFLAGS)
else
CXXFLAGS2  = $(CXXFLAGS) -O2 -march=native -Wall
CXXFLAGS  += -O3 -march=native -Wall
endif

ARCH      ?= sm_80
NVCCFLAGS  = -std=c++14 -arch=$(ARCH) -O2 -DSCALAR_TYPE=$(TYPE)

ifdef DEBUG
NVCCFLAGS += -g -G
endif

ALL = test.exe cuda_test.exe

default: help

all: $(ALL)

clean:
	@rm -rf *.o *.exe *.txt *.png *.csv *~

test: test.cpp
	@echo "Compiling CPU executable..."
	$(CXX) $(CXXFLAGS2) $< -I$(OPENBLAS_INC) -L$(OPENBLAS_LIB) $(OPENBLAS_FLAGS) -o $@

cuda_test.exe: cuda_test.cu
	@echo "Compiling CUDA executable..."
	$(NVCC) $(NVCCFLAGS) $< -o $@

help:
	@echo "Available targets :"
	@echo "    all              : compile CPU + CUDA"
	@echo "    test.exe         : CPU only"
	@echo "    cuda_test.exe    : CUDA only"
	@echo ""
	@echo "Options :"
	@echo "    DEBUG=yes        : mode debug"
	@echo "    TYPE=float|double: type scalar (default: float)"
	@echo "    ARCH=sm_XX       : architecture GPU (default: sm_80)"
	@echo ""
	@echo "Configuration :"
	@echo "    CXX       : $(CXX)"
	@echo "    CXXFLAGS  : $(CXXFLAGS2)"
	@echo "    NVCCFLAGS : $(NVCCFLAGS)"
	@echo "    OPENBLAS_INC   : $(OPENBLAS_INC)"
	@echo "    OPENBLAS_LIB   : $(OPENBLAS_LIB)"	
	@echo "    OPENBLAS_FLAGS : $(OPENBLAS_FLAGS)"
	@echo " Usage : "
	@echo "	make test.exe "                       
	@echo "	make test.exe TYPE=double"
	@echo " make cuda_test.exe ARCH=sm_86 "        
	@echo "	make all TYPE=double ARCH=sm_86"