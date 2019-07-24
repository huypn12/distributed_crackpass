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


LIB_OBJS=SHA1Cpu.o SHA1CpuSlice.o SHA1CpuKernel.o

all:SHA1Cpu.so SHA1Cpu.a Makefile

clean:
	rm -rf *.o *.so *.a kernel/*.o

SHA1Cpu.so:$(LIB_OBJS)
	$(LD) $(LDFLAGS) -o $@ $(LIB_OBJS) $(LIB_BOOST)

SHA1Cpu.a:$(LIB_OBJS)
	ar ru SHA1Cpu.a $(LIB_OBJS)

SHA1Cpu.o:${SRC_DIR}/SHA1Cpu.cpp ${INC_DIR}/SHA1Cpu.hpp
	$(CXX) $(CXXFLAGS) -I${INC_DIR} -o $@ -c $<

SHA1CpuSlice.o:${SRC_DIR}/SHA1CpuSlice.cpp ${INC_DIR}/SHA1CpuSlice.hpp
	$(CXX) $(CXXFLAGS) -I${INC_DIR} -o $@ -c $<

SHA1CpuKernel.o:${SRC_DIR}/kernel/SHA1CpuKernel.cpp ${SRC_DIR}/kernel/SHA1CpuKernel.hpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<
