CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall
OMPFLAGS = -fopenmp
NVCC = nvcc
# Use sm_75 for Turing+ GPUs, or sm_86 for Ampere+
# Use sm_52 for older Maxwell GPUs if needed
# CUDA 13.0 compatibility: force include cuda_fix.h and define constants
CUDAFLAGS = -O3 -arch=sm_75 -std=c++14 --compiler-options -include,$(INCDIR)/cuda_fix.h

# Define source and include directories
SRCDIR = src
INCDIR = include

# Add the include path to CXXFLAGS and CUDAFLAGS
# The -I flag tells the compiler to look in $(INCDIR) for header files
CXXFLAGS += -I$(INCDIR)
CUDAFLAGS += -I$(INCDIR)


# Week 1 targets
serial: $(SRCDIR)/main.cpp
	$(CXX) $(CXXFLAGS) -o ray_serial $(SRCDIR)/main.cpp

openmp: $(SRCDIR)/main.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o ray_openmp $(SRCDIR)/main.cpp

# Week 2 target (placeholder)
cuda: $(SRCDIR)/main_gpu.cu
	$(NVCC) $(CUDAFLAGS) -o ray_cuda $(SRCDIR)/main_gpu.cu

# Week 3 target (hybrid CPU-GPU)
hybrid: $(SRCDIR)/main_hybrid.cpp $(SRCDIR)/kernel.cu
	$(NVCC) $(CUDAFLAGS) -c $(SRCDIR)/kernel.cu
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -I/usr/local/cuda/include -c $(SRCDIR)/main_hybrid.cpp
	$(NVCC) $(CUDAFLAGS) -Xlinker -lgomp kernel.o main_hybrid.o -o ray_hybrid

clean:
	rm -f ray_serial ray_openmp ray_cuda ray_hybrid *.o *.ppm *.png

test: serial
	./ray_serial
	@if [ -f output_serial.ppm ]; then convert output_serial.ppm output_serial.png && echo "Created output_serial.png"; fi

test-openmp: openmp
	OMP_NUM_THREADS=4 ./ray_openmp
	@if [ -f output_openmp.ppm ]; then convert output_openmp.ppm output_openmp.png && echo "Created output_openmp.png"; fi

test-cuda: cuda
	./ray_cuda
	@if [ -f output_gpu.ppm ]; then convert output_gpu.ppm output_gpu.png && echo "Created output_gpu.png"; fi

test-hybrid: hybrid
	./ray_hybrid
	@if [ -f output_hybrid.ppm ]; then convert output_hybrid.ppm output_hybrid.png && echo "Created output_hybrid.png"; fi

benchmark: serial openmp cuda
	@echo "=== Performance Comparison ==="
	@echo -n "Serial: "; ./ray_serial | grep "Serial time"
	@echo -n "OpenMP: "; OMP_NUM_THREADS=4 ./ray_openmp | grep "OpenMP time"
	@echo -n "CUDA:   "; ./ray_cuda | grep "GPU rendering time"