NVCC = nvcc
OPENCV = `pkg-config opencv4 --cflags --libs`
CUDA_INCLUDES = -I/usr/local/cuda/include
CUDA_LIBS = -L/usr/local/cuda/lib64 -lcudart
NVCC_FLAGS = -O3 -arch=sm_60 -use_fast_math

all: Fuzzy_Edge_Detection_GPU

Fuzzy_Edge_Detection_GPU: Fuzzy_Edge_Detection_GPU.cu Fuzzy.cpp
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(OPENCV) $(CUDA_LIBS) $(CUDA_INCLUDES)

clean:
	rm -f Fuzzy_Edge_Detection_GPU *.o
