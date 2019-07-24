#!/bin/sh

#g++ MD5Crack.cpp -c -g -lboost_system -fPIC
#g++ MD5CpuSlice.cpp -c -g -lboost_system -lboost_thread -fPIC
#gcc -lstdc++ -lboost_system -fPIC -shared MD5Crack.o MD5CpuSlice.o -o MD5Crack.so
#g++ -g -Wall -fpic -lboost_system -lboost_thread -o md5 test_md5_crack.cpp MD5Crack.cpp MD5CpuSlice.cpp

nvcc -g -O3 -lboost_system -lboost_thread -D_DEBUG -I../include -ccbin g++ -o md5 main.cpp ../src/*.cu ../src/kernel/*.cu
