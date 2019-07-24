import logging
import json
import time
import threading

from task.master_task_md5crack import MasterTaskMD5Crack
from task.master_task_multimd5 import MasterTaskMultiMD5
from task.master_task_sha1crack import MasterTaskSHA1Crack

from model.node_model import NodeModel
from model.task_model import TaskModel


class MasterTaskManager(object):
    """ Master-side task manager.
    Consist of:
        (1) task information store
        (2) resource querying
        (3) resouce allocating for task
    """

    def __init__(self, agent):
        """ Constructor """
        self.controller = agent
        # Logger settings
        logging.basicConfig(filename='log/master-task-manager.log', level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        # Database connection.
        self.node_model = NodeModel()
        self.task_model = TaskModel()
        # Thread-safe mechanism
        self.is_running = False
        self.tm_lock = threading.Lock()
        # Working process: updating queue from db
        self.worker_process = threading.Thread(target=self.working_loop)
        self.worker_process.setDaemon(True)
        self.running_tasks = []


    ###########################################################
    ##-------------------- WORKING LOOP ---------------------##
    ###########################################################
    def launch(self, ):
        """ Activate worker process (launch working loop) """
        self.is_running = True
        self.worker_process.start()


    def finish(self, ):
        self.is_running = False
        self.worker_process.join()


    def working_loop(self, ):
        """ Working loop """
        while self.is_running:
            # Update running tasks' states to db.
            self.refresh_tasks()
            time.sleep(0.1)
            # Replenish task list;
            self.replenish_tasks()
            time.sleep(0.1)
            # Idling, avoid occupying CPU resources too frequently.


    def allocate_resources(self,):
        """
        @huypn
        Allocate resources
        @param  list : list of resources
        @param  list : candidate list
        @return dict : resource allocation to a task
        """
        pass

    ##@huypn
    def replenish_tasks(self, ):
        # @huypn: Acquire lock before performing actual calculation.
        # -- since Kombu is most likely multithreaded, this locking mechanism
        # -- is expectedly provide concurrency safety, with distributed rescs
        # -- on each workers are considered critical rescs.
        with self.tm_lock:
            # Query nodes ; get total amount of resources
            node_list = self.node_model.get_all_nodes()
            total_avail_cpu = 0
            total_avail_gpu = 0
            for node in node_list:
                total_avail_cpu += node['avail_cpu']
                total_avail_gpu += node['avail_gpu']
            cpu_prior_node_list = sorted(node_list,\
                                         key=lambda node: node['avail_cpu'])
            # Getting candidate tasks
            task_list = self.task_model.get_queued_tasks_by_arrived_time()
            if task_list == None or len(task_list) == 0:
                return
            # Choose a candidate task from list.
            print(total_avail_cpu)
            print(total_avail_gpu)
            candidate = None
            for i in range(0, len(task_list)):
                if task_list[i]['n_cpu'] <= total_avail_cpu and\
                        task_list[i]['n_gpu'] <= total_avail_gpu:
                    candidate = task_list[i]
                    break
            if candidate == None:
                print("Exit due to no candidate...")
                return
            # Logging
            print("DEBUG: Allocation scheme: totalcpu=%s, totalgpu=%s, candidatecpu=%s, candidategpu=%s"%(
                total_avail_cpu, total_avail_gpu, candidate['n_cpu'], candidate['n_gpu']
            ))
            # Assign resources:
            needed_cpu = candidate['n_cpu']
            needed_gpu = candidate['n_gpu']
            worker_list = []
            node_idx = 0
            while needed_cpu > 0 or needed_gpu > 0:
                # No need to check idx condition here
                # -- checked at preconditional test
                node = cpu_prior_node_list[node_idx]
                # Skip fully occupied node
                if node['avail_cpu'] == 0 and node['avail_gpu'] == 0:
                    node_idx += 1
                    continue
                # Calculating CPU allocation
                node_alloc_cpu = 0
                if needed_cpu > 0:
                    node_alloc_cpu = node['avail_cpu']
                    if needed_cpu <= node['avail_cpu']:
                        node_alloc_cpu = needed_cpu
                        node['avail_cpu'] -= needed_cpu
                        needed_cpu = 0
                    else:
                        needed_cpu -= node['avail_cpu']
                        node_alloc_cpu = node['avail_cpu']
                        node['avail_cpu'] = 0
                # Calculating GPU allocation
                node_alloc_gpu = 0
                if needed_gpu > 0:
                    node_alloc_gpu = node['avail_gpu']
                    if needed_gpu <= node['avail_gpu']:
                        node_alloc_gpu = needed_gpu
                        node['avail_gpu'] -= needed_gpu
                        needed_gpu = 0
                    else:
                        needed_gpu -= node['avail_gpu']
                        node_alloc_gpu = node['avail_gpu']
                        node['avail_gpu'] = 0
                # Skip null allocation
                if node_alloc_cpu == 0 and node_alloc_gpu == 0:
                    node_idx +=1
                    continue
                # Append calculated allocation to worker list
                worker = {}
                #@huypn: 2015-10-14 00:00 repair mistyped
                # worker['node_id'] = node_idx
                worker['node_id'] = node['node_id']
                worker['amqp_url'] = node['amqp_url']
                worker['n_cpu'] = node_alloc_cpu
                worker['n_gpu'] = node_alloc_gpu
                worker['avail_cpu'] = node['avail_cpu']
                worker['avail_gpu'] = node['avail_gpu']
                worker_list.append(worker)
                # Increase loop step
                node_idx += 1
            print("--> Allocation: %s"%str(worker_list))
            # Mixing arguments for MA communications to send into workers;
            task_alloc = {}
            task_alloc['task_id'] = candidate['task_id']
            task_alloc['type_name'] = candidate['type_name']
            task_alloc['params'] = json.loads(candidate['params'])
            task_alloc['worker_list'] = worker_list
            # Update node resources back into db
            for worker in worker_list:
                self.node_model.update_node_avail_resources({'node_id': worker['node_id'],\
                                                     'avail_cpu': worker['avail_cpu'],\
                                                     'avail_gpu': worker['avail_gpu']})
            # Update task status into db
            self.task_model.update_task_state({'task_id': candidate['task_id'],\
                                               'result': "",
                                               'new_state': 'IN_PROGRESS'})
            # Append task to in-mem running tasks list
            new_task = self.create_new_task({'task_id': candidate['task_id'],
                                             'type_name': candidate['type_name'],
                                             'params': json.loads(candidate['params'])})
            self.running_tasks.append(new_task)
            self.logger.info("Append candidate: %s"%(self.running_tasks,))
            # Send task creation mesg to all workers.
            mesg_body = {
                'task_id': candidate['task_id'],
                'type_name': candidate['type_name'],
                'n_cpu': 0,
                'n_gpu': 0,
                'params': json.loads(candidate['params'])
            }
            for worker in worker_list:
                mesg_body['n_cpu'] = worker['n_cpu']
                mesg_body['n_gpu'] = worker['n_gpu']
                self.controller.publish_mesg_create_task(worker['amqp_url'], mesg_body)


    #@huypn
    # TODO: change state autmata
    def refresh_tasks(self, ):
        """ Refresh tasks' state inside db;
        Do db-level: release resources;
        Must be launched before replenishment.
        """
        with self.tm_lock:
            for task in self.running_tasks:
                if task.current_state == 'TASK_STATE_FOUND_RESULT':
                    self.task_model.update_task_result({
                        'task_id': task.task_id,
                        'result': task.result_str
                    })
                elif task.current_state == 'FINISHED':
                    self.task_model.update_task_state({
                        'task_id': task.task_id,
                        "result": "not found",
                        'new_state': 'FINISHED'
                    })
                elif task.current_state == 'IN_PROGRESS':
                    full_progress = task.get_full_progress()
                    ratio_progress = full_progress['checked_size'] + '/' + full_progress['space_size']
                    eta = full_progress['eta']
                    self.task_model.update_task_progress({
                        'task_id': task.task_id,
                        'progress': ratio_progress,
                        'eta': eta
                    })
                else:
                    continue


    ###########################################################
    ##---------------- TASK MANAGEMENT  ---------------------##
    ###########################################################
    ##@huypn:
    def request_data_from_task(self, args):
        """ Trigger the task to popout a request
            Pull-model implementation requires that
            Master-side task popout a single chunk of data each time
            @param: args
            @return:
        """
        self.logger.info("[ Task manager ] <request_data_from_task> called args=%s"%(args,))
        try:
            result = None
            task_id = args['task_id']
            tasks = [t for t in self.running_tasks if t.task_id == task_id]
            if tasks is not None and len(tasks) > 0:
                task = tasks[0]
                result = task.pop_request(args)
                if result is not None:
                    result['task_id'] = task_id
                # else:
                    # self.update_completed_task({'task_id': task_id,
                                                # 'new_state': 'FINISHED'})
            return result
        except Exception as e:
            self.logger.error("[ Task manager ] <request_data_from_task> Exception:%s"%str(e))
            return -1


    ##@huypn:
    def push_result_to_task(self, args):
        """ Trigger the task to popout a request
            Pull-model implementation requires that
            Master-side task popout a single chunk of data each time
            @param: args
            @return:
        """
        self.logger.info("[ Task manager ] <push_result_to_task> called args=%s"%(args,))
        try:
            task_id = args['task_id']
            tasks = [t for t in self.running_tasks if t.task_id == task_id]
            if tasks is not None and len(tasks) > 0:
                task = tasks[0]
                task.push_result(args)
                self.update_completed_task({'task_id': task_id,
                                            'new_state': 'FOUND_RESULT'})
        except Exception as e:
            self.logger.error("[ Task manager ] <request_data_from_task> Exception:%s"%str(e))
            return -1


    ##@huypn:
    def create_new_task(self, args):
        """ As a task is appended to db from ma-fe;
            create task by request from ma,
            then append to running task list.
            FlowDirection < agent -> task_manager >
            @param: args
            @return:
        """
        try:
            type_name = args['type_name']
            if type_name == 'MD5':
                new_task = MasterTaskMD5Crack(args)
            elif type_name == 'MULTI_MD5':
                new_task = MasterTaskMultiMD5(args)
            elif type_name == 'SHA1':
                new_task = MasterTaskSHA1Crack(args)
            return new_task
        except Exception as e:
            self.logger.error("[ Task manager ] Exception:%s"%str(e))
            return -1


    ##@huypn:
    def shutdown_deactivated_task(self, args):
        """ Purge task from clusters and the task manager,
            Deactivated task is task requested to stop by user
            kill in clusters, while still maintain its info in db.
            @param: args {}
        """
        try:
            task_id = args['task_id']
            result = None
            for task in self.running_tasks:
                if task.task_id == task_id:
                    task.current_state = "DEACTIVATED"
                    task.active = False
            return result
        except Exception as e:
            self.logger.error("[ Task manager ] Exception:%s"%str(e))
            return -1

    ##@huypn:
    def update_completed_task(self, args):
        """ Completed task are tasks that reported finish by clusters.
            Store result; accept no more request; wait to be deactivated
            @param args
        """
        try:
            # Update task back into db
            args['new_state'] = 'FINISHED'
            self.task_model.update_task_state(args)
            # Publish kill task message
            # self.controller.publish_mesg_kill_task(args)
        except Exception as e:
            self.logger.error("[ Task manager ] Exception:%s"%str(e))
            return -1


    ###########################################################
    ##---------------- NODE MANAGEMENT  ---------------------##
    ###########################################################
    ##@huypn:
    def insert_node(self, args):
        """ Serves MasterAgent's on_register_mesg.
            Append nodes to running node list.
            @huypn2015-10-21T15:44:55: insert if does not exist
                -- else update
            param: args
            return:
        """
        node_id = self.node_model.insert_node(args)
        return node_id

    ##@huypn 2015-11-12T17:10:52
    def update_node(self, args):
        pass

    ##@huypn 2015-10-21T15:45:57
    def remove_node(self, args):
        """ Remove a node from list
        """
        pass

    ##@huypn 2015-10-21T15:46:34
    def retrieve_online_node(self, args):
        """
        """
        pass





###############################################
##---------------- Unit test ----------------##
###############################################
