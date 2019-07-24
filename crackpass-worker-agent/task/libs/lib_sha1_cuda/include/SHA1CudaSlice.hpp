#ifndef __SHA1_CUDA_SLICE__
#define __SHA1_CUDA_SLICE__


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


class SHA1Cuda;

class SHA1CudaSlice
{
    private:
        SHA1Cuda *fController;

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
        std::vector<unsigned char> fHashDigest;
        unsigned char *h_digest;
        unsigned char *d_digest;
        cudaTextureObject_t texDigest;

        std::string fCharsetStr;
        std::vector<unsigned char> fCharsetUChar;
        size_t fCharsetLen;
        unsigned char *h_charset;
        unsigned char *d_charset;
        cudaTextureObject_t texCharset;

        std::vector<unsigned char> fInputBase;
        size_t fInputLen;
        unsigned char *h_base;
        unsigned char *d_base;
        cudaTextureObject_t texBase;

        uint4 *h_kernelRes;
        uint4 *d_kernelRes;

        boost::thread *fWorkerThread;
        void WorkingLoop();

        void Init();
        int CallKernel(int requestOffset);
        void FreeCuda();

    public:
        SHA1CudaSlice( SHA1Cuda *controller );
        ~SHA1CudaSlice();

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

