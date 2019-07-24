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

// Standard C library
#include <cstdint>

// C++ STL
#include <string>
#include <vector>

/*
//@huypn:   using boost::multiprecision::cpp_int library to handle bigint
//          instead of @tannd's mybignum
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/thread/thread.hpp>
*/

#include <boost/thread/mutex.hpp>

#include "Vector.hpp"

class MultiMD5CpuSlice;

enum eTaskState {
    TASK_STATE_READY,
    TASK_STATE_RUNNING,
    TASK_STATE_STOPPED,
    TASK_STATE_WAITING_DATA,
    TASK_STATE_FOUND_RESULT,
    TASK_STATE_FINISHED
};
typedef enum eTaskState TaskState_t;

class MultiMD5Cpu
{
    private:
        boost::mutex fMutex;
        TaskState_t fTaskState; // 0: nothing; 1: found result; 2: request for more data;

        uint32_t fWorkSize;
        int32_t fCpuCores;
        std::vector<MultiMD5CpuSlice *> fCpuSliceList;

        std::string fCharsetStr;
        std::vector<unsigned char> fCharsetUChar;
        size_t fCharsetLen;

        std::vector<vec4uint32_t> fHashList;

        std::vector<unsigned char> fInputBase;
        size_t fInputLen;
        uint32_t fCurrentBaseIdx;

        std::vector<std::pair<std::string, std::string>> fResultList;
        // boost::multiprecision::cpp_int fStartIdx = 0;
        // boost::multiprecision::cpp_int fStopIdx = 0;

        void MD5digestToHex( std::string hashStr, std::vector<uint32_t> &htc );


    public:
        /**
         * Constructor
         * @param: int32_t      number of CPU threads
         * @param: const char*  hashes string
         * @param: const char*  charset string
         * @param:
         */
        MultiMD5Cpu(
                const int,
                const char **, const int,
                const char *
                );
        /**
         * Destructor
         * @param:
         */
        ~MultiMD5Cpu();

        /**
         * called from slices
         * @param
         * @param
         * @return
         */
        void CopyBasicParams( std::vector<vec4uint32_t> &, std::string & );
        /**
         * Get status of running task.
         * @param: void
         * @return: void
         */
        int GetState();
        /**
         * Pop request
         * @param: offset: int
         * @param: inputlen: uint32_t
         * @param: base: std::vector<uint>
         */
        bool PopRequest( std::vector<unsigned char> &, size_t&, uint32_t& );
        /** DLL_PUBLIC
         * Push request, c_string due to callability from C code
         * @param: start index: const char *
         * @param: stop index: const char *
         */
        void PushRequest( const char *, const size_t, const uint32_t );
        /** DLL_PUBLIC
         * Pop valid plaintext as c_str
         * @param: output cstring
         */
        int PopResult( char **result, char **hash );
        /**
         * Push result.
         * @param: plaintext: const char *
         */
        void PushResult( const std::string, const std::string );



        /**
         * Start running.
         */
        void Start();
        /**
         * Stop running.
         */
        void Stop();

};


#endif

