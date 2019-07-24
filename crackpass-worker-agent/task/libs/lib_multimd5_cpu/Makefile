

PROJECT=libMd5CrackCpu

COMMON_DIR=..
include $(COMMON_DIR)/Makefile.common

BIN_DIR=build
INC_DIR=include
SRC_DIR=src


LIB_OBJS=MultiMD5Cpu.o MultiMD5CpuSlice.o MultiMD5CpuKernel.o

all:MultiMD5Cpu.so MultiMD5Cpu.a Makefile

clean:
	rm -rf *.o *.so *.a kernel/*.o

MultiMD5Cpu.so:$(LIB_OBJS)
	$(LD) $(LDFLAGS) -o $@ $(LIB_OBJS) $(LIB_BOOST)

MultiMD5Cpu.a:$(LIB_OBJS)
	ar ru MultiMD5Cpu.a $(LIB_OBJS)

MD5Utils:${SRC_DIR}/MD5Utils.cpp ${INC_DIR}/MD5Utils.hpp
	$(CXX) $(CXXFLAGS) -I${INC_DIR} -o $@ -c $<

MultiMD5Cpu.o:${SRC_DIR}/MultiMD5Cpu.cpp ${INC_DIR}/MultiMD5Cpu.hpp
	$(CXX) $(CXXFLAGS) -I${INC_DIR} -o $@ -c $<

MultiMD5CpuSlice.o:${SRC_DIR}/MultiMD5CpuSlice.cpp ${INC_DIR}/MultiMD5CpuSlice.hpp
	$(CXX) $(CXXFLAGS) -I${INC_DIR} -o $@ -c $<

MultiMD5CpuKernel.o:${SRC_DIR}/kernel/MultiMD5CpuKernel.cpp ${INC_DIR}/MultiMD5CpuKernel.hpp
	$(CXX) $(CXXFLAGS) -I${INC_DIR} -o $@ -c $<
