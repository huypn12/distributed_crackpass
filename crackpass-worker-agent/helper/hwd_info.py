from helper.libs import hwd_info_stubs


def get_hwd_info():
    n_cpu = hwd_info_stubs.get_cpu_cores_count()
    n_gpu = hwd_info_stubs.get_gpu_cores_count()
    return {"total_cpu": n_cpu, "total_gpu": n_gpu}
