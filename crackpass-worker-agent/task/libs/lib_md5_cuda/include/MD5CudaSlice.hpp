#ifndef __MD5_CUDA_SLICE__
#define __MD5_CUDA_SLICE__


#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#include <boost/thread/thread.hpp>

enum eSlice {
    SLICE_STATE_READY,
    SLICE_STATE_RUNNING,
    SLICE_STATE_STOPPED,
    SLICE_STATE_WAITING_DATA,
    SLICE_STATE_FOUND_RESULT,
    SLICE_STATE_FINISHED
};
typedef enum eSlice eSliceState_t;


class MD5Cuda;

class MD5CudaSlice
{
    private:
        MD5Cuda *fController;

        eSliceState_t fSliceState;

        int fDeviceId;
        cudaDeviceProp fDeviceProps;

        int fBlockSize;
        int fGridSize;
        int fStreamCount;
        int fOffset;

        int fFoundResult;
        std::string fResultStr;

        std::string fHashStr;
        std::vector<unsigned int> fHashInts;
        uint4 fHashUint4;

        std::string fCharsetStr;
        std::vector<unsigned char> fCharsetUChar;
        size_t fCharsetLen;

        std::vector<unsigned char> fInputBase;
        size_t fInputLen;

        unsigned char *h_base;
        unsigned char *d_base;
        cudaTextureObject_t texBase;
        unsigned char *h_charset;
        unsigned char *d_charset;
        cudaTextureObject_t texCharset;
        uint4 *h_kernelRes;
        uint4 *d_kernelRes;

        boost::thread *fWorkerThread;
        void WorkingLoop();

        void Init();
        int CallKernel(int requestOffset);
        void FreeCuda();

    public:
        MD5CudaSlice( MD5Cuda *controller );
        ~MD5CudaSlice();

        /**
         * @param: device index: integer
         */
        void StartSlice( int );
        /**
         * @param:
         */
        void StopSlice();


};


#endif

