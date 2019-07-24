
#include <boost/multiprecision/cpp_int.hpp>

// \file    MD5CrackMaster.cpp
// \brief   implementation of masterside MD5 task\
//          serves as a data feeder
class MD5CrackMaster() {
    private:
        boost::multiprecision::cpp_int fCurrentBaseIdx;
        boost::multiprecision::cpp_int fSpaceSize;
        boost::multiprecision::cpp_int fCurrentPartialSize;

        std::string fCurrentBaseStr;
        std::string fCharsetStr;

        std::string fHashStr;
        std::vector< std::string > fHashStrVec;

    public:
        MD5CrackMaster(
                int nCpu, int nGpu,
                std::string charsetStr,
                std::vector< std::string > hashStrVec
                int minBaseLen, maxBaseLen
                );
        ~MD5CrackMaster();

        int PopRequest(

                );


}

