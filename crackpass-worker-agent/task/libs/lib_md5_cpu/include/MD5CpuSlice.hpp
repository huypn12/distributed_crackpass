#ifndef __MD5_CPU_SLICE__
#define __MD5_CPU_SLICE__

#include <string>
#include <vector>

#include <boost/thread/thread.hpp>


enum e_SliceState {
    SLICE_STATE_READY,
    SLICE_STATE_RUNNING,
    SLICE_STATE_STOPPED,
    SLICE_STATE_WAITING_DATA,
    SLICE_STATE_FOUND_RESULT,
};
typedef enum e_SliceState SliceState_t;

class MD5Cpu;

class MD5CpuSlice
{
    private:
        // Dependency Injection
        MD5Cpu *fController;

        SliceState_t fSliceState;
        // size of data chunk must be < ( 1<<8 -1 ), about 2.1 billion
        // (1 << 20) is recommended.
        uint32_t fWorksize;

        int32_t fFoundResult;
        std::string fResultStr;

        std::string fHashStr;
        std::vector<uint32_t> fHashInts;

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
         * @param: MD5Cpu: controller <- Dependency Injection
         * @return:
         */
        MD5CpuSlice( MD5Cpu *controller );
        /**
         * DESTRUCTOR
         * @param:
         * @return:
         */
        ~MD5CpuSlice();

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



