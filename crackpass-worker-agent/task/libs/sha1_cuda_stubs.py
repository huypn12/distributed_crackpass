from ctypes import *


# Try import the lib
#LIB_PATH = 'task/libs/lib_sha1_cuda/SHA1Cuda.so'
LIB_PATH = 'lib_sha1_cuda/SHA1Cuda.so'
LIB_SHA1_CUDA = None
try:
    LIB_SHA1_CUDA = cdll.LoadLibrary(LIB_PATH)
except Exception as expt:
    print("Error loading SHA1Cuda dll.")
    raise expt


def init(task_id, n_gpu, gpu_list, hash_str, charset_str):
    """init new cuda task"""
    f_init = LIB_SHA1_CUDA.SHA1CudaInit
    gpu_cptr = (c_int * len(gpu_list))(*gpu_list)
    f_init.argtypes = c_int, c_int, POINTER(c_int), c_char_p, c_char_p
    f_init.restype = c_int
    res = f_init(task_id, n_gpu, gpu_cptr, hash_str, charset_str)
    return res


def push_request(task_id, base_str, base_str_len, offset):
    """push data chunk"""
    f_push_request = LIB_SHA1_CUDA.SHA1CudaPushRequest
    f_push_request.argtypes = c_int, c_char_p, c_int, c_ulong
    f_push_request.restype = c_int
    res = f_push_request(task_id, base_str, base_str_len, offset)
    return res


def pop_result(task_id, result_str_lst):
    """# python string passed immutably makes """
    f_popresult = LIB_SHA1_CUDA.SHA1CudaPopResult
    result_buffer = create_string_buffer(128)
    res = f_popresult(task_id, result_buffer)
    result_str_lst.append(result_buffer)
    return res


def get_state(task_id):
    """get current task state"""
    f_getstate = LIB_SHA1_CUDA.SHA1CudaGetState
    f_getstate.argtypes = c_int
    f_getstate.restype = c_int
    res = f_getstate(task_id)
    return res


def start(task_id):
    """ activate """
    f_start = LIB_SHA1_CUDA.SHA1CudaStart
    f_start.argtypes = c_int
    f_start.restype = c_int
    res = f_start(task_id)
    return res


def stop(task_id):
    """deactiave"""
    f_stop = LIB_SHA1_CUDA.SHA1CudaStop
    f_stop.argtypes = c_int
    f_stop.restype = c_int
    res = f_stop(task_id)
    return res


