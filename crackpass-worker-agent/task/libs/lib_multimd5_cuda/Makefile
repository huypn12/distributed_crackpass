#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% @file: Makefile.common													 %%
#%% @desc: common compiler variables and options							 %%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

PROJECT=libMd5CrackCpu

COMMON_DIR=..
include $(COMMON_DIR)/Makefile.common

BIN_DIR=./build
INC_DIR=./include
SRC_DIR=./src


LIB_OBJS=MD5Cuda.o MD5CudaSlice.o MD5CudaKernel.o

all:MD5Cuda.so MD5Cuda.a Makefile

clean:
	rm -rf *.o *.so *.a kernel/*.o

MD5Cuda.so:$(LIB_OBJS)
	$(LD) $(LDFLAGS) -o $(BIN_DIR)/$@ $(LIB_OBJS) $(LIB_BOOST)

MD5Cuda.a:$(LIB_OBJS)
	ar ru MD5Cuda.a $(LIB_OBJS)

MD5Cuda.o:${SRC_DIR}/MD5Cuda.cu ${INC_DIR}/MD5Cuda.hpp
	$(NVCC) $(NVCCFLAGS) -I${INC_DIR} -o $@ -c $<

MD5CudaSlice.o:${SRC_DIR}/MD5CudaSlice.cu ${INC_DIR}/MD5CudaSlice.hpp
	$(NVCC) $(NVCCFLAGS) -I${INC_DIR} -o $@ -c $<

MD5CudaKernel.o:${SRC_DIR}/kernel/MD5CudaKernel.cu ${SRC_DIR}/kernel/MD5CudaKernel.hpp
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<
