from ctypes import *


class MD5CpuStubs(object):
    def __init__(self,):
        # Try import the lib
        lib_dir = 'task/libs/lib_md5_cpu/MD5Cpu.so'
        self.lib_md5_cpu = None
        try:
            self.lib_md5_cpu = cdll.LoadLibrary(lib_dir)
        except Exception as e:
            print( "Error loading MD5Cpu dll." )
            raise e

    def init(self, n_cpu, hash_str, charset_str ):
        f_init = self.lib_md5_cpu.MD5CpuInit
        f_init.argtypes = c_int, c_char_p, c_char_p
        f_init.restype = c_int
        res = f_init(n_cpu, hash_str, charset_str)
        return res


    def push_request(self, base_str, base_str_len, offset ):
        f_push_request = self.lib_md5_cpu.MD5CpuPushRequest
        f_push_request.argtypes = c_char_p, c_int, c_ulong
        f_push_request.restype = c_int
        res = f_push_request(base_str, base_str_len, offset)
        return res


    def pop_result(self, result_str_lst ):
        # python string passed immutably
        # makes list
        f_popresult = self.lib_md5_cpu.MD5CpuPopResult
        result_buffer = create_string_buffer(128)
        res = f_popresult(result_buffer)
        result_str_lst.append(result_buffer.value)
        return res


    def get_state(self, ):
        f_getstate = self.lib_md5_cpu.MD5CpuGetState
        f_getstate.restype = c_int
        res = f_getstate()
        return res


    def start(self, ):
        f_start = self.lib_md5_cpu.MD5CpuStart
        f_start.restype = c_int
        res = f_start()
        return res


    def stop(self, ):
        f_stop = self.lib_md5_cpu.MD5CpuStop
        f_stop.restype = c_int
        res = f_stop()
        return res


