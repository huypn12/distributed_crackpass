#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% @file: Makefile.common													 %%
#%% @desc: common compiler variables and options							 %%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

PROJECT=libMd5CrackCpu

COMMON_DIR=..
include $(COMMON_DIR)/Makefile.common

BIN_DIR=build
INC_DIR=include
SRC_DIR=src


LIB_OBJS=MD5Cpu.o MD5CpuSlice.o MD5CpuKernel.o

all:MD5Cpu.so MD5Cpu.a Makefile

clean:
	rm -rf *.o *.so *.a kernel/*.o

MD5Cpu.so:$(LIB_OBJS)
	$(LD) $(LDFLAGS) -o $@ $(LIB_OBJS) $(LIB_BOOST)

MD5Cpu.a:$(LIB_OBJS)
	ar ru MD5Cpu.a $(LIB_OBJS)

MD5Cpu.o:${SRC_DIR}/MD5Cpu.cpp ${INC_DIR}/MD5Cpu.hpp
	$(CXX) $(CXXFLAGS) -I${INC_DIR} -o $@ -c $<

MD5CpuSlice.o:${SRC_DIR}/MD5CpuSlice.cpp ${INC_DIR}/MD5CpuSlice.hpp
	$(CXX) $(CXXFLAGS) -I${INC_DIR} -o $@ -c $<

MD5CpuKernel.o:${SRC_DIR}/kernel/MD5CpuKernel.cpp ${SRC_DIR}/kernel/MD5CpuKernel.hpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<
