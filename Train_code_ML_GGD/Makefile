NVCC := /usr/local/cuda-9.0/bin/nvcc
#NVCC := nvcc
CC   := g++
NVCCFLAGS = -I/usr/local/cuda-9.0/include

TARGET = BPtrain_Sigmoid

ifeq ("$(MAKECMDGOALS)", "relu")
	NVCCFLAGs += -DRELU
	TARGET	   = BPtrain_Relu
endif

ifeq ("$(MAKECMDGOALS)", "sigmoid")
	NVCCFLAGs += -DSIGMOID
	TARGET	   = BPtrain_Sigmoid
endif

all:clean BPtrain_Sigmoid

sigmoid:BPtrain_Sigmoid
BPtrain_Sigmoid: clean BPtrain

relu:BPtrain_Relu
BPtrain_Relu: clean BPtrain

BPtrain: BPtrain.cc BP_GPU.o DevFunc.o Interface.o
	${NVCC}  $(NVCCFLAGS) -g BPtrain.cc BP_GPU.o DevFunc.o Interface.o -o $(TARGET) -L/usr/local/cuda-9.0/lib64 -lcublas -lcurand -lpthread
Interface.o: Interface.h Interface.cc
	${CC} $(NVCCFLAGS) -c  -g Interface.cc
DevFunc.o: DevFunc.h DevFunc.cu
	${NVCC} $(NVCCFLAGS) -c -g DevFunc.cu
BP_GPU.o: BP_GPU.h BP_GPU.cu DevFunc.o
	${NVCC} $(NVCCFLAGS) -c -g BP_GPU.cu


clean:
	rm -rf *.o
