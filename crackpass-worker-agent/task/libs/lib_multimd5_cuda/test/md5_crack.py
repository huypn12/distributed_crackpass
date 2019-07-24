from ctypes import *

import time


libcrack = cdll.LoadLibrary('./MD5Crack.so')

# Prepare parameters
args = {}
args['n_cpu'] = c_int(4)
args['n_gpu'] = c_int(1)
args['hash'] = c_char_p(b'72b77c145fadb106bb5d3b41bd4b46a5')
args['charset'] = c_char_p(b'abcd')

libcrack.MD5CrackInit(args['n_cpu'], args['n_gpu'], args['hash'], args['charset'])
libcrack.MD5CrackStart()
while True:
    print libcrack.MD5CrackGetState()
    base_str = 'aaaa'
    base_str_len = 4
    offset = 20
    f_md5crack_pushrequest = libcrack.MD5CrackPushRequest
    f_md5crack_pushrequest.argtypes = c_char_p, c_int, c_ulong
    f_md5crack_pushrequest.restype = c_int
    res = f_md5crack_pushrequest(base_str, base_str_len, offset)
    time.sleep(1)
# Try pushrequest
