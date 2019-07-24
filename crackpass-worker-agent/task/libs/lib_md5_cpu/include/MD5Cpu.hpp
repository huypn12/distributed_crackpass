#ifndef __MD5_CRACK_H__
#define __MD5_CRACK_H__


/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*+                 Shared Library Exporting Incantation                    +*/
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
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
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


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


enum e_TaskState {
    TASK_STATE_READY,
    TASK_STATE_RUNNING,
    TASK_STATE_STOPPED,
    TASK_STATE_WAITING_DATA,
    TASK_STATE_FOUND_RESULT,
    TASK_STATE_FINISHED
};
typedef enum e_TaskState TaskState_t;


class MD5CpuSlice;


class MD5Cpu
{
    private:
        boost::mutex fMutex;
        int fTaskId;
        TaskState_t fTaskState; // 0: nothing; 1: found result; 2: request for more data;

        int32_t fWorksize;
        int32_t fCpuCores;
        std::vector<MD5CpuSlice *> fCpuSliceList;

        std::string fCharsetStr;
        std::vector<unsigned char> fCharsetUChar;
        size_t fCharsetLen;

        std::string fHashStr;
        std::vector<uint32_t> fHashInts;

        std::vector<unsigned char> fInputBase;
        size_t fInputLen;
        int fCurrentBaseIdx;


        int fResultIdx;
        std::string fResultStr;
        // boost::multiprecision::cpp_int fStartIdx = 0;
        // boost::multiprecision::cpp_int fStopIdx = 0;

        /**
         * convert hash ascii string to vector repr.
         * @param: std::string      hash string
         * @param: std::vector<uint32_t>& hash vector
         * @return: void
         */
        void MD5digestToHex( std::string hashStr, std::vector<uint32_t> &htc );


    public:
        /**
         * Constructor
         * @param: int32_t      number of CPU threads
         * @param: const char*  hashes string
         * @param: const char*  charset string
         * @param:
         */
        MD5Cpu(int, const char *, const char *);
        /**
         * Destructor
         * @param:
         */
        ~MD5Cpu();


        /**
         * called from slices
         * @param
         * @param
         * @return
         */
        void CopyBasicParams( std::vector<uint32_t> &, std::string & );

        /**
         * Get status of running task.
         * @param: void
         * @return: void
         */
        int GetState();

        /**
         * Get task Id
         * @param:
         * @return: taskId
         */
        int GetTaskId();

        /**
         * Pop request
         * @param: offset: int
         * @param: inputlen: uint32_t
         * @param: base: std::vector<uint>
         */
        bool PopRequest( std::vector<unsigned char> &, size_t&, int & );

        /** DLL_PUBLIC
         * Push request, c_string due to callability from C code
         * @param: start index: const char *
         * @param: stop index: const char *
         */
        void PushRequest( const char *, const size_t, const int32_t );

        /** DLL_PUBLIC
         * Pop valid plaintext as c_str
         * @param: output cstring
         */
        int PopResult( char * );
        /**
         * Push result.
         * @param: plaintext: const char *
         */
        void PushResult( const std::string );
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

