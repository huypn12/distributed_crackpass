import json
import logging

# import lib.hwdinfo_stubs as hwdinfo_stubs
from helper import hwd_info

from task.md5_cpu import MD5Cpu
from task.md5_cuda import MD5Cuda
from task.multimd5_cpu import MultiMD5Cpu
from task.multimd5_cuda import MultiMD5Cuda
from task.sha1_cpu import SHA1Cpu
#from task.sha1_cuda import SHA1Cuda

class WorkerTaskManager(object):
    """
    Task management on the worker-side;
    corresponded with master-side implementation.
    """

    ###@huypn
    def __init__(self, agent):
        """ Constructor
        @param: agent
        """
        # Controller
        self.agent = agent
        # Hardware resources
        # hwdinfo = hwd_info.get_hwd_info()
        self.total_cpu = 8 # hwdinfo['total_cpu']
        self.total_gpu = 1 # hwdinfo['total_gpu']
        self.avail_cpu = self.total_cpu
        self.avail_gpu = self.total_gpu
        self.cuda_device_list = range(0, self.total_gpu)
        self.cuda_allocated_list = []
        # Local tasklist
        self.running_tasks = []
        # Logging settings
        logging.basicConfig(filename='/tmp/worker-task-manager.log', level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)


    ###########################################################
    ##------------------ Querying Node Info -----------------##
    ###########################################################
    def get_total_resources(self, ):
        """ Get total resources on the node """
        return {'total_cpu': self.total_cpu, 'total_gpu': self.total_gpu}


    def get_avail_resources(self, ):
        """ Get available (i.e unused) resources on the node """
        return {'avail_cpu': self.avail_cpu, 'avail_gpu': self.avail_gpu}


    def get_running_task(self, ):
        """ Get task list """
        return json.dumps(self.running_tasks)


    ###########################################################
    ##--------------- Serving WA Callbacks  -----------------##
    ###########################################################
    ##@huypn
    def has_avail_resources(self, ):
        """ check if there is anything for allocation """
        return self.avail_cpu > 0 and self.avail_gpu > 0


    def create_worker_task_md5(self, params):
        """ create md5 task"""
        params['n_gpu'] = 0
        if params['n_cpu'] > 0:
            task_local_id = len(self.running_tasks)
            params['task_local_id'] = task_local_id
            new_cpu_task = MD5Cpu(self, params)
            if new_cpu_task is None:
                self.logger.error("[ create_worker_task ] error occured.")
                return -1
            else:
                self.running_tasks.append(new_cpu_task)
                new_cpu_task.launch()
        if params['n_gpu'] > 0:
            task_local_id = len(self.running_tasks)
            params['task_local_id'] = task_local_id
            new_gpu_task = MD5Cuda(self, params)
            if new_gpu_task is None:
                self.logger.error("[ create_worker_task ] error occured.")
                return -1
            else:
                self.running_tasks.append(new_gpu_task)
                new_gpu_task.launch()
                return 0


    def create_worker_task_multimd5(self, params):
        """ create multi md5 task"""
        params['n_gpu'] = 0
        if params['n_cpu'] > 0:
            task_local_id = len(self.running_tasks)
            params['task_local_id'] = task_local_id
            new_cpu_task = MultiMD5Cpu(self, params)
            if new_cpu_task is None:
                self.logger.error("[ create_worker_task ] error occured.")
                return -1
            else:
                self.running_tasks.append(new_cpu_task)
                new_cpu_task.launch()
        if params['n_gpu'] > 0:
            task_local_id = len(self.running_tasks)
            params['task_local_id'] = task_local_id
            new_gpu_task = MultiMD5Cuda(self, params)
            if new_gpu_task is None:
                self.logger.error("[ create_worker_task ] error occured.")
                return -1
            else:
                self.running_tasks.append(new_gpu_task)
                new_gpu_task.launch()
                return 0


    def create_worker_task_sha1(self, params):
        """ create sha1 task"""
        params['n_gpu'] = 0
        if params['n_cpu'] > 0:
            task_local_id = len(self.running_tasks)
            params['task_local_id'] = task_local_id
            new_cpu_task = SHA1Cpu(self, params)
            if new_cpu_task is None:
                self.logger.error("[ create_worker_task ] error occured.")
                return -1
            else:
                self.running_tasks.append(new_cpu_task)
                new_cpu_task.launch()
        if params['n_gpu'] > 0:
            task_local_id = len(self.running_tasks)
            params['task_local_id'] = task_local_id
            new_gpu_task = SHA1Cuda(self, params)
            if new_gpu_task is None:
                self.logger.error("[ create_worker_task ] error occured.")
                return -1
            else:
                self.running_tasks.append(new_gpu_task)
                new_gpu_task.launch()


    def create_worker_task(self, params):
        """ create new task instance
        TODO: throw exceptions """
        # Precondition check
        ret_code = 0
        ret_mesg = ""
        if not self.has_avail_resources():
            ret_mesg = "[create_worker_task] no avail resources"
            self.logger.error(ret_mesg, " params=%s", params)
            ret_code = -1
        if params['n_cpu'] == 0 and params['n_gpu'] == 0:
            ret_mesg = "[create_worker_task] invalid alloc request"
            self.logger.error(ret_mesg, " params=%s", params)
            ret_code = -1
        if ret_code == -1:
            # precondition check failed
            return ret_code
        ########## MD5 task ##########
        if params['type_name'] == 'MD5':
            ret_code = self.create_worker_task_md5(params)
        ########## Multi MD5 task ##########
        elif params['type_name'] == 'MULTI_MD5':
            ret_code = self.create_worker_task_multimd5(params)
        ########## SHA1 task ##########
        elif params['type_name'] == 'SHA1':
            ret_code = self.create_worker_task_sha1(params)
        else:
            # invalid task type
            ret_mesg = "[create_worker_task] invalid task type."
            self.logger.error(ret_mesg, " params=", params)
            ret_code = -1
        if ret_code == -1:
            # postcondition check failed
            return ret_code


    def push_request_to_task(self, args):
        """push data chunk into task to process"""
        task_id = args['task_id']
        task_local_id = args['task_local_id']
        tasks = [t for t in self.running_tasks \
                 if t.task_id == task_id and \
                 t.task_local_id == task_local_id]
        if tasks is not None and len(tasks) > 0:
            task = tasks[0]
            task.push_request(args)
        else:
            return -1


    def kill_worker_task(self, params):
        """ Force task to stop its work. Remove from running tasks."""
        task_id = params['task_id']
        tasks = [t for t in self.running_tasks if t.task_id == task_id]
        if tasks is not None and len(tasks) > 0:
            task = tasks[0]
            self.running_tasks.remove(task)
        else:
            return -1



    ###########################################################
    ##--------------- Serving Task Pulling  -----------------##
    ###########################################################
    ##@huypn
    def report_pull_request(self, args):
        """handling data request, report to agent"""
        print("[ Task manager ] Requested data from %s.%s"%(args['task_id'], args['task_local_id']))
        try:
            self.agent.publish_mesg_request_data(args)
        except Exception as expt:
            print("pull request error, ", str(expt))
            raise expt

    ##@huypn
    def report_push_result(self, args):
        """handling result reporting"""
        print("[ Task manager ] Result from %s.%s"%(args['task_id'], args['task_local_id']))
        try:
            self.agent.publish_mesg_push_result(args)
        except Exception as expt:
            print("pull request error, ", str(expt))
            raise expt

    ##@huypn
    def report_finished_task(self, task_id, result):
        print("finished task %s"%task_id)
        try:
            args = {'task_id': task_id, 'result': result}
            self.agent.publish_mesg_finished_task(args)
        except Exception as expt:
            print("finishing request error, ", str(expt))
            raise expt



