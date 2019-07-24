#ifndef __MD5_CPU_SLICE__
#define __MD5_CPU_SLICE__

#include <string>
#include <vector>

#include <boost/thread/thread.hpp>


enum eSliceState {
    SLICE_STATE_READY,
    SLICE_STATE_RUNNING,
    SLICE_STATE_STOPPED,
    SLICE_STATE_WAITING_DATA,
    SLICE_STATE_FOUND_RESULT,
};
typedef enum eSliceState SliceState_t;

#include "SHA1Cpu.hpp"

class SHA1Cpu;

class SHA1CpuSlice
{
    private:
        // Dependency Injection
        SHA1Cpu *fController;

        SliceState_t fSliceState;
        // size of data chunk must be < ( 1<<8 -1 ), about 2.1 billion
        // (1 << 20) is recommended.
        uint32_t fWorksize;

        int32_t fFoundResult;
        std::string fResultStr;

        std::string fHashStr;
        std::vector<uint8_t> fHashDigest;

        std::string fCharsetStr;
        std::vector<unsigned char> fCharsetUChar;
        size_t fCharsetLen;

        std::vector<unsigned char> fInputBase;
        size_t fInputLen;

        /**
         * Main working loop
         * @param: void
         * @return: void
         */
        void WorkingLoop();
        boost::thread *fWorkerThread = NULL;

    public:
        /**
         * CONSTRUCTOR
         * @param: SHA1Cpu: controller <- Dependency Injection
         * @return:
         */
        SHA1CpuSlice( SHA1Cpu *controller );
        /**
         * DESTRUCTOR
         * @param:
         * @return:
         */
        ~SHA1CpuSlice();

        /**
         * Start running CPU Instance
         * @param:
         * @return:
         */
        void StartSlice();
        /**
         * Stop running CPU Instance
         * @param:
         * @return:
         */
        void StopSlice();
};



#endif



