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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>


int DLL_PUBLIC print_psz_str( const char **list, const int n_elems )
{

    assert( list != NULL );

    for( int i = 0; i < n_elems; i++ )
    {
        printf("String: %s\n", list[i]);
    }

    return 0;

}

int DLL_PUBLIC pop_psz_str( char *result )
{
     char *something = "fuckyou";
     strncpy(result, something, 128);
    result[128] = 0;
     /**result = something;*/
     printf("Copied: %s\n", result);
     return 0;
}
