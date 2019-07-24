from ctypes import *


try:
    psz_str_lib = cdll.LoadLibrary("./pszStr.so")
except Exception as e:
    raise e

def print_psz_str():
    str_list = ['abc', 'def']
    str_list_p = (c_char_p * len(str_list))()
    str_list_p[:] = str_list
    func_ptr = psz_str_lib.print_psz_str
    res = func_ptr(str_list_p, len(str_list))
    print res

def pop_psz_str():
    result_str = create_string_buffer(128)
    func_ptr = psz_str_lib.pop_psz_str
    res = func_ptr(result_str)
    print result_str.value


if __name__ == '__main__':
    print_psz_str()
    pop_psz_str()
