#ifndef __MD5_CPU_SLICE__
#define __MD5_CPU_SLICE__

#include <cstdint>

#include <string>
#include <vector>

#include <boost/thread/thread.hpp>


enum eSliceState {
    SLICE_STATE_READY,
    SLICE_STATE_RUNNING,
    SLICE_STATE_STOPPED,
    SLICE_STATE_WAITING_DATA,
    SLICE_STATE_FOUND_RESULT,
    SLICE_STATE_FINISHED
};
typedef enum eSliceState SliceState_t;

class MultiMD5Cpu;

class MultiMD5CpuSlice
{
    private:
        // Dependency Injection
        MultiMD5Cpu *fController;

        SliceState_t fSliceState;
        // size of data chunk must be < ( 1<<8 -1 ), about 2.1 billion
        // (1 << 20) is recommended.
        uint32_t fWorkSize;

        std::vector<vec4uint32_t> fResultList;

        std::vector<vec4uint32_t> fHashList;

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
         * @param: MultiMD5Cpu: controller <- Dependency Injection
         * @return:
         */
        MultiMD5CpuSlice( MultiMD5Cpu *controller );
        /**
         * DESTRUCTOR
         * @param:
         * @return:
         */
        ~MultiMD5CpuSlice();

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



