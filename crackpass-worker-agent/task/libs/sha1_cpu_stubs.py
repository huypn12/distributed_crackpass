""" load shared lib: sha1_cpu """

from ctypes import *


# Try import the lib
LIB_PATH = 'task/libs/lib_sha1_cpu/SHA1Cpu.so'
# LIB_PATH = 'lib_sha1_cpu/SHA1Cpu.so'
DLL_SHA1CPU = None
try:
    DLL_SHA1CPU = cdll.LoadLibrary(LIB_PATH)
except Exception as expt:
    print("Error loading sha1Cpu dll.")
    raise expt


def init(task_id, n_cpu, hash_str, charset_str):
    """ init new task """
    f_init = DLL_SHA1CPU.SHA1CpuInit
    f_init.argtypes = c_int, c_int, c_char_p, c_char_p
    f_init.restype = c_int
    res = f_init(task_id, n_cpu, hash_str, charset_str)
    return res


def push_request(task_id, base_str, base_str_len, offset):
    """ push data bucket to task """
    f_push_request = DLL_SHA1CPU.SHA1CpuPushRequest
    f_push_request.argtypes = c_int, c_char_p, c_int, c_ulong
    f_push_request.restype = c_int
    res = f_push_request(task_id, base_str, base_str_len, offset)
    return res


def pop_result(task_id, result_str_lst):
    """ Pop result if found """
    f_popresult = DLL_SHA1CPU.SHA1CpuPopResult
    result_buffer = create_string_buffer(128)
    res = f_popresult(task_id, result_buffer)
    result_str_lst.append(result_buffer.value)
    return res


def get_state(task_id):
    """ get task state """
    f_getstate = DLL_SHA1CPU.SHA1CpuGetState
    f_getstate.restype = c_int
    res = f_getstate(task_id)
    return res


def start(task_id):
    """ start task """
    f_start = DLL_SHA1CPU.SHA1CpuStart
    f_start.restype = c_int
    res = f_start(task_id)
    return res


def stop(task_id):
    """ stop task """
    f_stop = DLL_SHA1CPU.sha1CpuStop
    f_stop.restype = c_int
    res = f_stop(task_id)
    return res


