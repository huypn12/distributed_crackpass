from task.libs import sha1_cpu_stubs


def test_task_6char():
    task_id = 1
    sha1_cpu_stubs.init(task_id,\
                        32,\
                        # "4b4f65f1ae6a72e8a48c605527bad951f791fa56",     # sontay
                        # "55387c2ba86f676cca7d3d4aec1bf5ce7088b61c",     # noibai
                        # "787d0f35e7495bdb393cfc484ae5ce6a57852378",       # bacson
                        "9df8da34ccaadaf833f4d786b97ec97fc1506ebb",       # hadong
                        "abcdefghijklmnopqrstuvwxyz")
    sha1_cpu_stubs.start(task_id)
    sha1_cpu_stubs.push_request(task_id, "aaaaaa", 6, 26**6)
    while sha1_cpu_stubs.get_state(task_id) != 4:
        continue
    result_str_lst = []
    sha1_cpu_stubs.pop_result(task_id, result_str_lst)
    print(result_str_lst)


def test_task_5char():
    task_id = 1
    sha1_cpu_stubs.init(task_id,\
                        32,\
                        "2aec56b6f154c2ff8f2c63d00bdb09d675943679",     # hanoi
                        # "3abfcc786884db91ecd14509dc62fad0da2e3d76",     # hanam
                        # "d6971690dbf480ea773110da946b505db3067042",     # hatay
                        # "506bb846fc02369b69d4f17cc7f351f803a85614",     # annam
                        "abcdefghijklmnopqrstuvwxyz")
    sha1_cpu_stubs.start(task_id)
    sha1_cpu_stubs.push_request(task_id, "aaaaa", 5, 26**5)
    while sha1_cpu_stubs.get_state(task_id) != 4:
        continue
    result_str_lst = []
    sha1_cpu_stubs.pop_result(task_id, result_str_lst)
    print(result_str_lst)


def test_task_4char():
    task_id = 2
    sha1_cpu_stubs.init(task_id,\
                        32,\
                        # "6c0c231fa004af781d28ec5be037bc01f89cb008",     # sapa
                        # "a34e034b42fd16e1577ea79c8c47fa3d00551da9",     # dong
                        # "99d16e88e62df038555277704b0e8d37ba2aa9de",     # tuan
                        "b29025a795c4f5ea28fd2d7270a5600fe8099077",     # minh
                        "abcdefghijklmnopqrstuvwxyz")
    sha1_cpu_stubs.start(task_id)
    sha1_cpu_stubs.push_request(task_id, "aaaa", 4, 26**4)
    while sha1_cpu_stubs.get_state(task_id) != 4:
        continue
    result_str_lst = []
    sha1_cpu_stubs.pop_result(task_id, result_str_lst)
    print(result_str_lst)


import threading
if __name__ == '__main__':
    t1 = threading.Thread(target=test_task_6char)
    t2 = threading.Thread(target=test_task_5char)
    t3 = threading.Thread(target=test_task_4char)
    t1.start()
    t2.start()
    t3.start()
