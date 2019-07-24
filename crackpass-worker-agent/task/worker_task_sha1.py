import time
import threading

#from libs import sha1_cuda_stubs
from libs import sha1_cpu_stubs
from worker_task import WorkerTask


class WorkerTaskSHA1(WorkerTask):

    """ Worker-side implementation of a task.
    Corresponded with master-side implementation
    """

    ##@huypn
    def __init__(self, controller, params):
        """
        @param: controller: class TaskManager
        @param: task_id: int
        @param: n_cpu: int
        @param: n_gpu: int
        """
        # Init superclass
        # super.__init__(controller)
        # Management id
        self.task_id = params['task_id']
        # Dependency injection
        self.controller = controller
        self.task_local_id = params['task_local_id']
        self.params = params['params']
        # Upper level waiting flag
        self.is_waiting = False
        # Upper level running flag, serving forced exit
        self.is_running = False
        # CPU thread
        self.occupied_cpu = params['n_cpu']
        self.worker_thread = threading.Thread(target=self.working_loop_cpu)
        self.worker_thread.setDaemon(True)
        # CUDA thread
        self.worker_thread_cuda = params['gpu_list']
        self.worker_thread = threading.Thread(target=self.working_loop_cuda)
        self.worker_thread.setDaemon(True)


    ##@huypn:
    def working_loop_cpu(self, ):
        """ Working loop for the worker thread.
        Backward calling from C code to Python is rudimentary.
        The task representation must keep probing the down-level lib.
        @param self -- this
        """
        #--- init
        sha1_cpu_stubs.init(self.task_id,
                            self.occupied_cpu,
                            self.params['hash_str'],
                            self.params['charset_str'])
        sha1_cpu_stubs.start(self.task_id)
        suggested_offset = 1 << 20
        self.controller.report_pull_request({'task_id': self.task_id,
                                             'task_local_id': self.task_local_id,
                                             'offset': suggested_offset})
        self.task_state = sha1_cuda_stubs.get_state(self.task_id)
        #--- main loop
        while self.task_state != WorkerTask.C_TASK_STATE['TASK_STATE_STOPPED']      and \
                self.task_state != WorkerTask.C_TASK_STATE['TASK_STATE_FINISHED']   and \
                self.is_running == True:
            # Probing running process
            self.task_state = self.sha1_cuda_stubs.get_state(self.task_id)
            # Waiting for data
            if self.task_state == WorkerTask.C_TASK_STATE['TASK_STATE_WAITING_DATA'] or \
                    self.task_state == WorkerTask.C_TASK_STATE['TASK_STATE_RUNNING'] or \
                    self.is_waiting == True:
                time.sleep(0.1)
                continue
            # Result found
            elif self.task_state == WorkerTask.C_TASK_STATE['TASK_STATE_FOUND_RESULT']:
                # Get the fucking result
                res_str_lst = []
                self.sha1_cuda_stubs.pop_result(res_str_lst)
                # Report it to master
                self.controller.report_finished_task()
                # Change current state.
                self.is_running = False
                # Idle, wait to be killed
                time.sleep(0.1)
            # Requires pushing data
            elif self.task_state == WorkerTask.C_TASK_STATE['TASK_STATE_WAITING_DATA']:
                self.is_waiting = True
                suggested_offset = self.occupied_cpu * (1 << 24)
                self.controller.report_pull_request({'task_id': self.task_id,
                                                     'task_local_id': self.task_local_id,
                                                     'offset': suggested_offset})
        #clean exit
        self.sha1_cuda_stubs.stop()


    ##@huypn:
    def working_loop_cuda(self, ):
        """ Working loop for the worker thread.
        Backward calling from C code to Python is rudimentary.
        The task representation must keep probing the down-level lib.
        @param self -- this
        """
        #--- init
        sha1_cuda_stubs.init(self.task_id,
                            self.occupied_cpu,
                            self.params['hash_str'],
                            self.params['charset_str'])
        self.sha1_cuda_stubs.start(self.task_id)
        suggested_offset = 1 << 20
        self.controller.report_pull_request({'task_id': self.task_id,
                                             'task_local_id': self.task_local_id,
                                             'offset': suggested_offset})
        self.task_state = self.sha1_cuda_stubs.get_state(self.task_id)
        #--- main loop
        while self.task_state != WorkerTask.C_TASK_STATE['TASK_STATE_STOPPED']      and \
                self.task_state != WorkerTask.C_TASK_STATE['TASK_STATE_FINISHED']   and \
                self.is_running == True:
            # Probing running process
            self.task_state = self.sha1_cuda_stubs.get_state(self.task_id)
            # Waiting for data
            if self.task_state == WorkerTask.C_TASK_STATE['TASK_STATE_WAITING_DATA'] or \
                    self.task_state == WorkerTask.C_TASK_STATE['TASK_STATE_RUNNING'] or \
                    self.is_waiting == True:
                time.sleep(0.1)
                continue
            # Result found
            elif self.task_state == WorkerTask.C_TASK_STATE['TASK_STATE_FOUND_RESULT']:
                # Get the fucking result
                res_str_lst = []
                self.sha1_cuda_stubs.pop_result(res_str_lst)
                # Report it to master
                self.controller.report_finished_task()
                # Change current state.
                self.is_running = False
                # Idle, wait to be killed
                time.sleep(0.1)
            # Requires pushing data
            elif self.task_state == WorkerTask.C_TASK_STATE['TASK_STATE_WAITING_DATA']:
                self.is_waiting = True
                suggested_offset = self.occupied_cpu * (1 << 24)
                self.controller.report_pull_request({'task_id': self.task_id,
                                                     'task_local_id': self.task_local_id,
                                                     'offset': suggested_offset})
        #clean exit
        self.sha1_cuda_stubs.stop()


    ## Push request
    def push_request(self, args):
        """ Push received response data to task.
        Flow directive: controller->task
        """
        try:
            base_str = args['base_str']
            base_len = args['base_str_len']
            offset = args['offset']
            self.sha1_stubs.push_request(base_str, base_len, offset)
            self.is_waiting = False
        except Exception as e:
            raise e

    ##@huypn:
    def launch(self, ):
        """ Start running loop.
        @param:
        """
        self.is_running = True
        self.worker_thread.start()


    ##@huypn
    def finish(self, ):
        """ Close running loop, join the thread.
        @param:
        """
        self.is_running = False
        self.worker_thread.join()

