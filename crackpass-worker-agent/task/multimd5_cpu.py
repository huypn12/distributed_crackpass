import time
import threading

from libs import md5_cpu_stubs

from worker_task import WorkerTask


class MultiMD5Cpu(WorkerTask):

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
        #super().__init__()
        # Management id
        self.controller = controller
        self.task_id = params['task_id']
        self.task_local_id = 0
        self.params = params['params']
        # Upper level waiting flag
        self.is_waiting = False
        # Upper level running flag, serving forced exit
        self.is_running = False
        # CPU thread
        self.occupied_cpu = params['n_cpu']
        self.worker_thread = threading.Thread(target=self.working_loop)
        self.worker_thread.setDaemon(True)


    ##@huypn:
    def working_loop(self, ):
        """ Working loop for the worker thread.
        Backward calling from C code to Python is rudimentary.
        The task representation must keep probing the down-level lib.
        @param self -- this
        """
        md5_cpu_stubs.init(self.occupied_cpu,
                           self.params['hash_str'],
                           self.params['charset_str'])
        self.md5_cpu_stubs.start()
        suggested_offset = 1 << 28
        self.controller.report_pull_request({'task_id': self.task_id,
                                             'offset': suggested_offset})
        self.task_state = self.md5_cpu_stubs.get_state()
        while self.task_state != WorkerTask.C_TASK_STATE['TASK_STATE_STOPPED']      and \
                self.task_state != WorkerTask.C_TASK_STATE['TASK_STATE_FINISHED']   and \
                self.is_running == True:
            # Probing running process
            self.task_state = self.md5_cpu_stubs.get_state()
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
                self.md5_cpu_stubs.pop_result(res_str_lst)
                # Report it to master
                self.controller.report_finished_task()
                # Change current state.
                self.is_running = False
                # Idle, wait to be killed
                time.sleep(0.1)
            # Requires pushing data
            elif self.task_state == WorkerTask.C_TASK_STATE['TASK_STATE_READY']:
                self.is_waiting = True
                suggested_offset = self.occupied_cpu * (1 << 24)
                self.controller.report_pull_request({'task_id': self.task_id,
                                                     'task_local_id': self.task_local_id,
                                                     'offset': suggested_offset})
        #clean exit
        self.md5_cpu_stubs.stop()


    ## Push request
    def push_request(self, args):
        """ Push received response data to task.
        Flow directive: controller->task
        """
        self.is_waiting = False
        try:
            base_str = args['base_str']
            base_len = args['base_str_len']
            offset = args['offset']
            self.md5_stubs.push_request(base_str, base_len, offset)
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

