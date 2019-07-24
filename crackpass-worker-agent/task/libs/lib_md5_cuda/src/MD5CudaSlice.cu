#include <iostream>

#include <cstdio>
#include <cstdlib>

#include "MD5Cuda.hpp"
#include "MD5CudaSlice.hpp"
#include "kernel/MD5CudaKernel.hpp"


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// => For debugging purposes
//

//@huypn
/*
   Catching Cuda Errors
 */
static void HandleError(
        cudaError_t err,
        const char *file,
        int line
        )
{
    if (err != cudaSuccess)
    {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/**
  printing base
  */
#ifdef _DEBUG
void printBase(
        std::vector<unsigned char> base,
        std::string charset,
        size_t baseLen
        )
{
    std::cout << "szBase = ";
    for( int i = 0; i < baseLen; i++ )
    {
        std::cout << charset[base[i]];
    }
    std::cout << std::endl;
}
#endif


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MD5CudaSlice::MD5CudaSlice(
        MD5Cuda *controller
        )
{
    fController = controller;
    fSliceState = SLICE_STATE_READY;

    // init cuda params
    fDeviceId = -1;
    fBlockSize = 0;
    fGridSize = 0;
    fStreamCount = 0;
    fOffset = 0;

    //@huypn: copying running parameters
    fController->CopyBasicParams( fHashInts, fCharsetStr );
    fHashUint4.x = fHashInts[0];
    fHashUint4.y = fHashInts[1];
    fHashUint4.z = fHashInts[2];
    fHashUint4.w = fHashInts[3];
#ifdef _DEBUG
    std::cout << "digest X=" << std::hex << fHashUint4.x << std::endl;
    std::cout << "digest Y=" << std::hex << fHashUint4.y << std::endl;
    std::cout << "digest Z=" << std::hex << fHashUint4.z << std::endl;
    std::cout << "digest W=" << std::hex << fHashUint4.w << std::endl;
#endif

    // Taking care of charset
    fCharsetLen = fCharsetStr.size();
    for( int i = 0; i < fCharsetLen; i++ ) {
        fCharsetUChar.push_back( (unsigned char) fCharsetStr[i] );
    }
#ifdef _DEBUG
    for( int i = 0; i < fCharsetLen; i++ ) {
        std::cout << fCharsetUChar[i];
    }
    std::cout << std::endl;
#endif

    fInputBase.resize(16);
    std::fill(fInputBase.begin(), fInputBase.end(), 0);

}


MD5CudaSlice::~MD5CudaSlice()
{
    // Finalize work
    StopSlice();
    delete fWorkerThread;
}


void MD5CudaSlice::FreeCuda() {
    // Sync device
    cudaDeviceSynchronize();
    // Destroy texture object
    cudaDestroyTextureObject(texBase);
    cudaDestroyTextureObject(texCharset);
    // Free CUDA malloc 
    cudaFree(d_kernelRes);
    cudaFree(d_charset);
    cudaFree(d_base);
    // Reset Cuda Device
    cudaDeviceReset();
}

void MD5CudaSlice::StartSlice( int deviceId )
{
    fSliceState = SLICE_STATE_RUNNING;
    fDeviceId = deviceId;
    fWorkerThread = new boost::thread( boost::bind(&MD5CudaSlice::WorkingLoop, this) );
}

void MD5CudaSlice::StopSlice()
{
    fSliceState = SLICE_STATE_STOPPED;
    cudaDeviceSynchronize();
    fWorkerThread->join();
}


void MD5CudaSlice::Init()
{
    HANDLE_ERROR( cudaSetDevice( fDeviceId ) );
    HANDLE_ERROR( cudaGetDeviceProperties( &fDeviceProps, fDeviceId ));

    //#CUDA_OCCUPANCY_API: Get blocksize and gridsize
    int blockSize = 0;
    int minGridSize = 0;
    HANDLE_ERROR( cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)MD5CudaKernel, 0, 0) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    std::cout << "OCCUPANCY APIs: blocksize=" << blockSize << ";gridSize=" << minGridSize << std::endl;
    fBlockSize = blockSize;
    fGridSize = minGridSize;
    fStreamCount = 1;
    fOffset = fBlockSize * fGridSize * fStreamCount;

    // emergency flag: set when result found;
    h_kernelRes = (uint4 *) malloc(sizeof(uint4));
    *h_kernelRes = make_uint4(0,0,0,0);
    HANDLE_ERROR( cudaMalloc((void**)&d_kernelRes, sizeof(uint4)) );
    HANDLE_ERROR( cudaMemcpy(d_kernelRes, h_kernelRes, sizeof(uint4), cudaMemcpyHostToDevice) );

    // Allocate input base, load into mapped memory
    // Not like charset, base could be resized :-s <-fuckit, no need
    HANDLE_ERROR( cudaMalloc( (void**)&d_base, 16*sizeof(unsigned char)) );
    HANDLE_ERROR( cudaMemcpy( d_base, &fInputBase[0], 16*sizeof(unsigned char), cudaMemcpyHostToDevice) );
    cudaResourceDesc resDescBase;
    memset( &resDescBase, 0, sizeof(resDescBase) );
    resDescBase.resType = cudaResourceTypeLinear;
    resDescBase.res.linear.devPtr = d_base;
    resDescBase.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDescBase.res.linear.desc.x = 8;
    resDescBase.res.linear.sizeInBytes = 16 * sizeof( unsigned char );
    cudaTextureDesc texDescBase;
    memset(&texDescBase, 0, sizeof(texDescBase));
    texDescBase.readMode = cudaReadModeElementType;
    texBase = 0;
    HANDLE_ERROR( cudaCreateTextureObject( &texBase, &resDescBase, &texDescBase, NULL ) );


    // #CUDA_BINDLESS_TEXTURE
    // Allocate charset; load into bindless texture 
    HANDLE_ERROR( cudaMalloc((void**)&d_charset, sizeof(unsigned char)*fCharsetLen) );
    HANDLE_ERROR( cudaMemcpy(d_charset, &fCharsetUChar[0], sizeof(unsigned char)*fCharsetLen, cudaMemcpyHostToDevice) );
    cudaResourceDesc resDescCharset;
    memset( &resDescCharset, 0, sizeof(resDescCharset) );
    resDescCharset.resType = cudaResourceTypeLinear;
    resDescCharset.res.linear.devPtr = d_charset;
    resDescCharset.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDescCharset.res.linear.desc.x = 8;
    resDescCharset.res.linear.sizeInBytes = fCharsetLen*sizeof( unsigned char );
    cudaTextureDesc texDescCharset;
    memset(&texDescCharset, 0, sizeof(texDescCharset));
    texDescCharset.readMode = cudaReadModeElementType;
    texCharset=0;
    HANDLE_ERROR( cudaCreateTextureObject( &texCharset, &resDescCharset, &texDescCharset, NULL ) );
}


// Kernel wrapper
int MD5CudaSlice::CallKernel( int worksize )
{
    // Setting up launching parameter
    int blockSize = fBlockSize;
    int gridSize = (worksize + blockSize - 1) / blockSize;
    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid(gridSize, 1, 1);
    MD5CudaKernel<<<dimGrid, dimBlock>>>(
            fHashUint4, d_kernelRes,
            texBase, fInputLen,
            texCharset , fCharsetLen,
            worksize
            );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    int ret = -1;
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        ret = -2;
    } else {
        ret = -1;
        cudaMemcpy(h_kernelRes, d_kernelRes, sizeof(uint4), cudaMemcpyDeviceToHost);
        if( h_kernelRes->y == 1 ) {
             ret = 0;
        } 
    }
    return ret;
}


void MD5CudaSlice::WorkingLoop()
{
    Init();
    while( fSliceState != SLICE_STATE_STOPPED &&
            fSliceState != SLICE_STATE_FINISHED ) {
        // Prepare data.
        int requestOffset = fOffset;
        bool hasData = fController->PopRequest(
                fInputBase, fInputLen, requestOffset );
        if (!hasData) {
            // Avoid CPU consuming when idle;
            boost::this_thread::sleep(
                    boost::posix_time::milliseconds(50));
            continue;
        }
        // Update base for new kernel call
        HANDLE_ERROR( cudaMemcpy( d_base, &fInputBase[0], 16*sizeof(unsigned char), cudaMemcpyHostToDevice) );
        // Call kernel
        int ret = CallKernel(requestOffset);
        // CUDA error -> dung pha ma
        if( ret == -2 ) {
            FreeCuda();
            return;
        } else if( ret != -1 ) {
            uint32_t counter = h_kernelRes->x;
            uint32_t i = fInputLen;
            for (uint32_t j = 0, a = 0, carry = 0;
                    j < i;
                    ++j, counter /= fCharsetLen)
            {
                a = fInputBase[j] + carry + counter % fCharsetLen;
                if (a >= fCharsetLen) {
                    carry = 1;
                    a -= fCharsetLen;
                }
                else carry = 0;
                fResultStr.push_back( fCharsetStr[a] );
            }
            fController->PushResult( fResultStr );
            std::cout << "Found result: " << fResultStr <<std::endl;
            std::cout << "Debugging digest: A=" << std::hex <<h_kernelRes->z <<
                ";B=" << std::hex << h_kernelRes->w << std::endl;
            return;
        }
    }
}



