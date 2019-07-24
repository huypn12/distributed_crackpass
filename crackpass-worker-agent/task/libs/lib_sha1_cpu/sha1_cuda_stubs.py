from ctypes import *


# Try import the lib
lib_dir = 'task/libs/md5_cuda/MD5Cuda.so'
lib_md5_cuda = None
try:
    lib_md5_cuda = cdll.LoadLibrary(lib_dir)
except Exception as e:
    print( "Error loading MD5Cuda dll." )
    raise e


def init( n_gpu, gpu_list, hash_str, charset_str ):
    f_init = lib_md5_cuda.MD5CudaInit
    gpu_cptr = (c_int * len(gpu_list))(*gpu_list)
    f_init.argtypes = c_int, c_int_p, c_char_p, c_char_p
    f_init.restype = c_int
    res = f_init(n_gpu, gpu_cptr, hash_str, charset_str)
    return res


def push_request( base_str, base_str_len, offset ):
    f_push_request = lib_md5_cuda.MD5CudaPushRequest
    f_push_request.argtypes = c_char_p, c_int, c_ulong
    f_push_request.restype = c_int
    res = f_push_request(base_str, base_str_len, offset)
    return res


def pop_result( result_str_lst ):
    # python string passed immutably
    # makes list
    f_popresult = lib_md5_cuda.MD5CudaPopResult
    result_buffer = create_string_buffer(128)
    res = f_popresult(result_buffer)
    result_str_lst.append(result_str)
    return res


def get_state():
    f_getstate = lib_md5_cuda.MD5CudaGetState
    f_getstate.restype = c_int
    res = f_getstate()
    return res


def start():
    f_start = lib_md5_cuda.MD5CudaStart
    f_start.restype = c_int
    res = f_start()
    return res


def stop():
    f_stop = lib_md5_cuda.MD5CudaStop
    f_stop.restype = c_int
    res = f_stop()
    return res


