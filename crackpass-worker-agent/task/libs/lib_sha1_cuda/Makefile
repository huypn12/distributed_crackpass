#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% @file: Makefile.common													 %%
#%% @desc: common compiler variables and options							 %%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

PROJECT=libSha1CrackCuda

COMMON_DIR=..
include $(COMMON_DIR)/Makefile.common

BIN_DIR=./build
INC_DIR=./include
SRC_DIR=./src


LIB_OBJS=SHA1Cuda.o SHA1CudaSlice.o SHA1CudaKernel.o

all:SHA1Cuda.so SHA1Cuda.a Makefile

clean:
	rm -rf *.o *.so *.a kernel/*.o

SHA1Cuda.so:$(LIB_OBJS) SHA1CudaLinked.o
	$(LD) $(LDFLAGS) -o $@ $(LIB_OBJS) SHA1CudaLinked.o $(LIB_BOOST) $(LIB_CUDA)

SHA1Cuda.a:$(LIB_OBJS)
	ar ru SHA1Cuda.a $(LIB_OBJS)

SHA1CudaLinked.o:$(LIB_OBJS)
	$(NVCC) $(NVCCFLAGS) -dlink $(LIB_OBJS) -o SHA1CudaLinked.o

SHA1Cuda.o:${SRC_DIR}/SHA1Cuda.cu ${INC_DIR}/SHA1Cuda.hpp
	$(NVCC) $(NVCCFLAGS) -I${INC_DIR} -o $@ -c $<

SHA1CudaSlice.o:${SRC_DIR}/SHA1CudaSlice.cu ${INC_DIR}/SHA1CudaSlice.hpp
	$(NVCC) $(NVCCFLAGS) -I${INC_DIR} -o $@ -c $<

SHA1CudaKernel.o:${SRC_DIR}/kernel/SHA1CudaKernel.cu ${SRC_DIR}/kernel/SHA1CudaKernel.hpp
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<
