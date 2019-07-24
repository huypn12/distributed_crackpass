#### @file: Makefile
#### @desc: Make file for cpuinfo
PROJECT=libGpuInfo

COMMON_DIR=../
include $(COMMON_DIR)/Makefile.common

BIN_DIR=build
INC_DIR=include
SRC_DIR=src

all:GpuInfo.so GpuInfo.a
	cp $(BIN_DIR)/GpuInfo.so ./

GpuInfo.so:GpuInfo.o
	$(LD) $(LDFLAGS) ${BIN_DIR}/$< -o $(BIN_DIR)/$@ $(LIB_CUDA) $(LIB_BOOST) 

GpuInfo.a:GpuInfo.o
	$(RANLIB) $(BIN_DIR)/$@ ${BIN_DIR}/$<

GpuInfo.o:$(SRC_DIR)/GpuInfo.cu $(INC_DIR)/GpuInfo.cuh
	$(NVCC) $(NVCCFLAGS) -I${INC_DIR} -dc $< -o ${BIN_DIR}/_$@
	$(NVCC) $(NVCCFLAGS) -I${INC_DIR} -dlink ${BIN_DIR}/_$@ -o ${BIN_DIR}/$@ -lcudart

clean:
	rm -rf ${BIN_DIR}/*
