#ifndef __MD5_CRACK_H__
#define __MD5_CRACK_H__


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


class MD5CudaSlice;


class MD5Cuda
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
        std::vector<unsigned int> fHashInts;

        std::vector<unsigned char> fInputBase;
        size_t fInputLen;

        int fOffset;
        int fCurrentBaseIdx;

        std::string fResultStr;

        void GetIntsFromHash( std::string hashStr, std::vector<unsigned int> &htc );

        std::vector<MD5CudaSlice *> fCudaSliceList;

    public:
        /**
         * Constructor
         * @param: ncpu: int
         * @param: ncudalist: int*
         * @param: hash string
         * @param: charset string
         */
        MD5Cuda( int, int*, const char *, const char *);
        /**
         * Destructor
         * @param:
         */
        ~MD5Cuda();

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
        void CopyBasicParams( std::vector<unsigned int> &, std::string & );
        void MD5digestToHex( std::string, std::vector<unsigned int> &);

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
class MD5CudaSlice {
    private:
        MD5Cuda *fController;
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
        MD5CudaSlice(MD5Cuda *controller) {
            fController = controller;
            fIsRunning = true;
            fWorkingThread = new std::thread( &MD5CudaSlice::WorkingLoop, this );
        }

        void Suspend() {
            fIsRunning = false;
            fWorkingThread->join();
        }
};

class MD5CpuSlice {
    private:
        MD5Cuda *fController;
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
        MD5CpuSlice(MD5Cuda *controller) {
            fController = controller;
            fIsRunning = true;
            fWorkingThread = new std::thread( &MD5CpuSlice::WorkingLoop, this );
        }

        void Suspend() {
            fIsRunning = false;
            fWorkingThread->join();
        }
};
**/

