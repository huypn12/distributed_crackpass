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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

DLL_LOCAL struct my_obj {
    int id;
};

typedef struct my_obj my_obj_t;

static my_obj_t *static_obj_list[1024 ];
static size_t curr_obj_idx = 0;

static my_obj_t *dynamic_obj = NULL;


void DLL_PUBLIC init()
{
    if (static_obj_list == NULL) {
        printf("No instance pool.\n");
    } else {
        printf("Static object pool exists at <%d>.\n", static_obj_list);
        my_obj_t *static_obj = (my_obj_t *) malloc(sizeof(my_obj_t));
        if (static_obj != NULL) {
            printf("New object instance at <%d>\n", static_obj);
            static_obj->id = 2 * curr_obj_idx;
            static_obj_list[curr_obj_idx] = static_obj;
            curr_obj_idx++;
        }
    }

    if ( dynamic_obj == NULL ) {
        printf("NULL dynamic object, create a new one at");
        dynamic_obj = (my_obj_t *) malloc(sizeof(my_obj_t));
        if (dynamic_obj != NULL) {
            printf("<%d>\n", dynamic_obj);
        }
    } else {
        printf("Dynamic object exists at <%d>.\n", dynamic_obj);
    }
}

void DLL_PUBLIC check()
{
    for (int i = 0; i < curr_obj_idx; i++) {
        if (static_obj_list[i]->id == 2) {
            printf("Found object at <%d>\n", static_obj_list[i]);
        }
    }
}
