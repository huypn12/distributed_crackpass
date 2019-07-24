
cd ..
make DEBUG=1
cd test
g++ -D_DEBUG -std=c++11 -lboost_system -lboost_thread -O3 -pipe -Wall -fPIC main.cpp ../MD5Cpu.a
