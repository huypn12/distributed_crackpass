import threading
import ctypes


def working_loop():
    # initialization
    lib_path = "./test_lib.so"
    lib_dll = None
    try:
        lib_dll = ctypes.cdll.LoadLibrary(lib_path)
    except Exception as e:
        raise e
    init_func_ptr = lib_dll.init
    init_func_ptr()
    check_func_ptr = lib_dll.check
    check_func_ptr()
    # main


if __name__ == '__main__':
    thread1 = threading.Thread(name="t_1", target=working_loop)
    thread2 = threading.Thread(name="t_2", target=working_loop)
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()


