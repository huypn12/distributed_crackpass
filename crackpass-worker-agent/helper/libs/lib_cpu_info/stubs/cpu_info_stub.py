from ctypes import *


# Try import the lib
lib_dir = 'lib/MD5Cpu.so'
lib_md5_crack = None
try:
    lib_md5_crack_cpu = cdll.LoadLibrary(lib_dir)
except Exception as e:
    raise e

def init( n_cpu, hash_str, charset_str ):
    f_init = lib_md5_crack_cpu.MD5CpuInit
    f_init.argtypes = c_int, c_int, c_char_p, c_char_p
    f_init.restype = c_int
    res = f_init(n_cpu, n_gpu, hash_str, charset_str)
    return res


def push_request( base_str, base_str_len, offset ):
    f_push_request = lib_md5_crack_cpu.MD5CpuPushRequest
    f_push_request.argtypes = c_char_p, c_int, c_ulong
    f_push_request.restype = c_int
    res = f_push_request(base_str, base_str_len, offset)
    return res


def pop_result( result_str_lst ):
    f_popresult = lib_md5_crack_cpu.MD5CpuPopResult
    f_popresult.argtypes = POINTER(c_char_p) # char **
    f_popresult.restype = c_int
    result_str = ""
    res = f_popresult(result_str) # !!!Potentially problematic!!! -> python string passed immutably
    result_str_lst.append(result_str)
    return res


def get_state():
    f_getstate = lib_md5_crack_cpu.MD5CpuGetState
    f_getstate.restype = c_int
    res = f_getstate()
    return res


def start():
    f_start = lib_md5_crack_cpu.MD5CpuStart
    f_start.restype = c_int
    res = f_start()
    return res


def stop():
    f_stop = lib_md5_crack_cpu.MD5CpuStop
    f_stop.restype = c_int
    res = f_stop()
    return res


