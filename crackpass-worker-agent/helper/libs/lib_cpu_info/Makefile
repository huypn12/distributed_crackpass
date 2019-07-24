#### @file: Makefile
#### @desc: Make file for cpuinfo
PROJECT=libCpuInfo

COMMON_DIR=../
include $(COMMON_DIR)/Makefile.common 

BIN_DIR=build
INC_DIR=include
SRC_DIR=src

all:CpuInfo.so CpuInfo.a

CpuInfo.so:CpuInfo.o
	$(LD) $(LDFLAGS) $(LIB_BOOST) ${BIN_DIR}/$< -o $(BIN_DIR)/$@
	cp $(BIN_DIR)/$@ ./

CpuInfo.a:CpuInfo.o
	$(RANLIB) $(BIN_DIR)/$@ ${BIN_DIR}/$< 

CpuInfo.o:$(SRC_DIR)/CpuInfo.cpp $(INC_DIR)/CpuInfo.hpp
	$(CXX) $(CXXFLAGS) -I${INC_DIR} -c $< -o ${BIN_DIR}/$@

clean:
	rm -rf ${BIN_DIR}/*
