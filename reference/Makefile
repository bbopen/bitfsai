NVCC ?= nvcc
ARCH ?= native

.PHONY: all check check-host clean

all: test_bitfsai standalone_amg

test_bitfsai: test_bitfsai.cu bitfsai.cuh certified_damping.hpp row_projection.hpp workloads.hpp
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) test_bitfsai.cu -o test_bitfsai

standalone_amg: standalone_amg.cu workloads.hpp solver_checks.hpp row_projection.hpp
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) --extended-lambda standalone_amg.cu -o standalone_amg

test_row_projection: test_row_projection.cpp row_projection.hpp
	$(CXX) -O3 -std=c++17 test_row_projection.cpp -o test_row_projection

test_certified_damping: test_certified_damping.cpp certified_damping.hpp
	$(CXX) -O2 -std=c++17 -fno-fast-math -frounding-math test_certified_damping.cpp -o test_certified_damping

check-host: test_row_projection test_certified_damping
	./test_row_projection
	./test_certified_damping

check: check-host test_bitfsai standalone_amg
	./test_bitfsai
	./standalone_amg --self-test

clean:
	rm -f test_bitfsai standalone_amg test_row_projection test_certified_damping
