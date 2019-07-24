#ifndef __SHA1_CRACK_H__
#define __SHA1_CRACK_H__


/*---------- DLL export incatantion ----------*/
#if defined _WIN32 || defined __CYGWIN__
#  ifdef BUILDING_DLL
#    ifdef __GNUC__
#      define DLL_PUBLIC __attribute__((dllexport))
#    else
#      define DLL_PUBLIC __declspec(dllexport)
#    endif
#  else
#    ifdef __GNUC__
#      define DLL_PUBLIC __attribute__((dllimport))
#    else
#      define DLL_PUBLIC __declspec(dllimport)
#    endif
#    define DLL_LOCAL
#  endif
#else
#  if __GNUC__ >= 4
#    define DLL_PUBLIC __attribute__ ((visibility("default")))
#    define DLL_LOCAL  __attribute__ ((visibility("hidden")))
#  else
#    define DLL_PUBLIC
#    define DLL_LOCAL
#  endif
#endif
/*--------------------------------------------*/

#include <string>
#include <vector>

/*
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/thread/thread.hpp>
*/
#include <boost/thread/mutex.hpp>

enum e_TaskState {
    TASK_STATE_READY,
    TASK_STATE_RUNNING,
    TASK_STATE_STOPPED,
    TASK_STATE_WAITING_DATA,
    TASK_STATE_FOUND_RESULT,
    TASK_STATE_FINISHED
};
typedef enum e_TaskState TaskState_t;


class SHA1CudaSlice;


class SHA1Cuda
{
    private:
        TaskState_t fTaskState;

        boost::mutex fMutex;

        int fCudaDeviceCount;
        std::vector<int> fCudaDeviceList;

        std::string fCharsetStr;
        std::vector<unsigned char> fCharsetUChar;
        size_t fCharsetLen;

        std::string fHashStr;
        std::vector<unsigned char> fHashDigest;

        std::vector<unsigned char> fInputBase;
        size_t fInputLen;

        int fOffset;
        int fCurrentBaseIdx;

        std::string fResultStr;

        void GetIntsFromHash( std::string hashStr, std::vector<unsigned int> &htc );

        std::vector<SHA1CudaSlice *> fCudaSliceList;

    public:
        /**
         * Constructor
         * @param: ncpu: int
         * @param: ncudalist: int*
         * @param: hash string
         * @param: charset string
         */
        SHA1Cuda( int, int*, const char *, const char *);
        /**
         * Destructor
         * @param:
         */
        ~SHA1Cuda();

        /**
         * Pop request
         * @param: offset: int
         * @param: inputlen: unsigned int
         * @param: base: std::vector<uint>
         */
        bool PopRequest( std::vector<unsigned char> &, size_t&, int& );
        /**
         * Push result.
         * @param: plaintext: const char *
         */
        void PushResult( const std::string );
        /**
         * called from slices
         * @param
         * @param
         * @return
         */
        void CopyBasicParams( std::vector<unsigned char> &, std::string & );
        void SHA1strToDigest( std::string, std::vector<unsigned char> &);

        //----------------------------------------------
        //----------- Callable from C dll---------------
        /**
         * Push request, c_string due to callability from C code
         * @param: start index: const char *
         * @param: stop index: const char *
         */
        void PushRequest( const char *, const unsigned int, const int );
        /**
         *@param: output cstring
         */
        int PopResult( char * );
        /**
         * Get status of running task.
         */
        int GetState();
        /**
         * Start running.
         */
        void Start();
        /**
         * Stop running.
         */
        void Stop();
        //----------------------------------------------
        //----------------------------------------------

};


#endif





































/**
#include <iostream>
// Fucking dummy test
class SHA1CudaSlice {
    private:
        SHA1Cuda *fController;
        boost::multiprecision::cpp_int fBase = 0;
        bool fIsRunning = false;
        std::thread *fWorkingThread = NULL;
        void WorkingLoop() {
            while( fIsRunning ) {
                std::this_thread::sleep_for( std::chrono::milliseconds(1000) );
                fController->PopRequest( fBase, 1000 );
                std::cout << fBase << std::endl;
            }
        }

    public:
        SHA1CudaSlice(SHA1Cuda *controller) {
            fController = controller;
            fIsRunning = true;
            fWorkingThread = new std::thread( &SHA1CudaSlice::WorkingLoop, this );
        }

        void Suspend() {
            fIsRunning = false;
            fWorkingThread->join();
        }
};

class SHA1CpuSlice {
    private:
        SHA1Cuda *fController;
        boost::multiprecision::cpp_int fBase = 0;
        bool fIsRunning = false;
        std::thread *fWorkingThread = NULL;
        void WorkingLoop() {
            while( fIsRunning ) {
                std::this_thread::sleep_for( std::chrono::milliseconds(1000) );
                fController->PopRequest( fBase, 1000 );
            }
        }

    public:
        SHA1CpuSlice(SHA1Cuda *controller) {
            fController = controller;
            fIsRunning = true;
            fWorkingThread = new std::thread( &SHA1CpuSlice::WorkingLoop, this );
        }

        void Suspend() {
            fIsRunning = false;
            fWorkingThread->join();
        }
};
**/

