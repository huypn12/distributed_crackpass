import logging
import json

from kombu import Connection, Queue, Exchange
from kombu.mixins import ConsumerMixin
from kombu.pools import producers

from config import amqp_config
from worker_task_manager import WorkerTaskManager


class WorkerAgent(ConsumerMixin):
    """ Worker agent, this thing must be daemon-ized.
    Communication methods are wrapped within.
    Communication separating in to another class is unecessarily complication.
    At least for now.
    """

    def __init__(self, conn):
        # Setting up logger
        logging.basicConfig(filename='log/worker-agent.log', level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        # Create local task manager
        self.worker_task_manager = None#WorkerTaskManager(self)
        # self identification
        self.name = '' # Assigned on register
        self.amqp_url = ''# Assigned on register
        # Establish connection to master
        self.master_url = '' # must be filled with dynamic configuration
        self.master_conn = None #Connection(self.master_url)
        # Amqp connection args
        self.connection = conn
        self.exchange = Exchange('e_worker', 'direct', durable=True)
        # Declare queues
        self.q_create_task = Queue('q_create_task',
                                   exchange=self.exchange,
                                   routing_key='rk_create_task')
        self.q_kill_task = Queue('q_kill_task',
                                 exchange=self.exchange,
                                 routing_key='rk_kill_task')
        self.q_response_data = Queue('q_response_data',
                                     exchange=self.exchange,
                                     routing_key='rk_response_data')
        ## Not used yet.
        self.q_query_info = Queue('q_query_info',
                                  exchange=self.exchange,
                                  routing_key='rk_query_info')


    ###########################################################
    ##---------------- Utility functions --------------------##
    ###########################################################
    def activate_task_manager(self, ):
        """ Activate worker-side taskmanager """
        self.worker_task_manager = WorkerTaskManager(self)


    def establish_master_conn(self, args):
        """ Establish connection to master """
        try:
            self.master_url = args['master_url']
            self.master_conn = Connection(self.master_url)
        except Exception as e:
            print("Error establishing connection to master.")
            raise e


    def get_total_resources(self, ):
        return self.worker_task_manager.get_total_resources()

    ###########################################################
    ##------------------ Mesg Receiving ---------------------##
    ###########################################################
    ##@huypn:
    def on_mesg_create_task(self, body, mesg):
        """ Create message received.
            -- Create a new task, based on args.
            @param: args
            @return:
        """
        print("< Q_CREATE_TASK > Message received body=%s"%(body,))
        self.logger.info("< Q_CREATE_TASK > Message received: %s"%(body,))
        # Catch Error case
        #@huypn 2015-11-11T22:40:16
        # -> Must be replaced with error queue
        if body is None:
            self.logger.error("< Q_CREATE_TASK > None message received %s"%str(mesg))
            return
        elif body == -1:
            self.logger.error("< Q_CREATE_TASK > Error message received %s"%str(mesg))
            return
        elif body['n_cpu'] == 0 and body['n_gpu'] == 0:
            print("Mesg received, do nothing")
            return
        else:
            pass
        # Do the actual job
        try:
            self.worker_task_manager.create_worker_task(body)
        except Exception as e:
            self.logger.error("Exception occurred: %s"%(str(e),))
        finally:
            mesg.ack()


    ##@huypn:
    def on_mesg_kill_task(self, body, mesg):
        """ Request killing task from master.
        -- Cluster side > actuall killing in task manager.
        @param: args
        @return: None
        """
        print("< Q_KILL_TASK > Message received")
        self.logger.info("< Q_KILL_TASK > Message received: %s"%(body,))
        # Catch Error case
        #@huypn 2015-11-11T22:40:16
        # -> Must be replaced with error queue
        if body is None:
            self.logger.error("< Q_KILL_TASK > None message received %s"%str(mesg))
            return
        elif body == -1:
            self.logger.error("< Q_KILL_TASK > Error message received %s"%str(mesg))
            return
        else:
            pass
        # Do the actual job
        try:
            self.worker_task_manager.kill_worker_task(body)
        except Exception as e:
            self.logger.error("Exception occurred: %s"%(str(e),))
        finally:
            mesg.ack()


    def on_mesg_response_data(self, body, mesg):
        """ Push received response data to task
        @param: args
        @return: None
        """
        print("< Q_RESPONSE_DATA > Message received")
        self.logger.info("< Q_RESPONSE_DATA > Message received: %s"%(body,))
        # Catch Error case
        #@huypn 2015-11-11T22:40:16
        # -> Must be replaced with error queue
        if body is None:
            self.logger.error("< Q_RESPONSE_DATA > None message received %s"%str(mesg))
            return
        elif body == -1:
            self.logger.error("< Q_RESPONSE_DATA > Error message received %s"%str(mesg))
            return
        else:
            pass
        # Do the actual job
        try:
            result = self.worker_task_manager.push_request_to_task(body)
            print("Pushed request to task, res = ",result)
        except Exception as e:
            self.logger.error("Exception occurred: %s"%(str(e),))
        finally:
            mesg.ack()


    def get_consumers(self, Consumer, channel):
        return [
            Consumer(self.q_create_task,
                     callbacks=[self.on_mesg_create_task],
                     accept=['json']),
            Consumer(self.q_kill_task,
                     callbacks=[self.on_mesg_kill_task],
                     accept=['json']),
            Consumer(self.q_response_data,
                     callbacks=[self.on_mesg_response_data],
                     accept=['json'])
        ]


    ###########################################################
    ##---------------- Mesg Publishing  ---------------------##
    ###########################################################
    def publish_mesg_register_node(self, args):
        """ call on initialization, send node registration to master """
        print("< publisher > publish register mesg")
        try:
            self.name = args['name']
            self.amqp_url = args['amqp_url']
            with producers[self.master_conn].acquire(block=True) as producer:
                ex = Exchange("e_master", type="direct")
                rk = 'rk_register'
                producer.publish(
                    args,
                    routing_key=rk,
                    exchange=ex,
                    serializer='json'
                )
        except Exception as e:
            self.logger.error("publish_remote: %s"%(str(e),))


    def publish_mesg_request_data(self, args):
        """ Send message to q_create_task of node_id """
        print("< publisher > publish request data mesg")
        try:
            # Signing sender
            args['name'] = self.name
            args['amqp_url'] = self.amqp_url
            with producers[self.master_conn].acquire(block=True) as producer:
                ex = Exchange("e_master", type="direct")
                rk = 'rk_request'
                producer.publish(args,
                                 exchange=ex,
                                 routing_key=rk,
                                 serializer='json')
        except Exception as e:
            self.logger.error("publish_remote: %s"%(str(e),))


    def publish_mesg_push_result(self, args):
        """ Send message to q_finished_task of node_id """
        print("< publisher > publish finished task mesg")
        try:
            args['name'] = self.name
            args['amqp_url'] = self.amqp_url
            with producers[self.master_conn].acquire(block=True) as producer:
                ex = Exchange("e_master", type="direct")
                rk = 'rk_finished'
                producer.publish(
                    json.dumps(args),
                    exchange=ex,
                    routing_key=rk,
                    serializer='json'
                )
        except Exception as e:
            self.logger.error("publish_remote: %s"%(str(e),))


    def publish_mesg_finished_task(self, args):
        """ Send message to q_finished_task of node_id """
        print("< publisher > publish finished task mesg")
        try:
            args['name'] = self.name
            args['amqp_url'] = self.amqp_url
            with producers[self.master_conn].acquire(block=True) as producer:
                ex = Exchange("e_master", type="direct")
                rk = 'rk_finished'
                producer.publish(
                    args,
                    exchange=ex,
                    routing_key=rk,
                    serializer='json'
                )
        except Exception as e:
            self.logger.error("publish_remote: %s"%(str(e),))




###########################################################
##---------------- Activating agent ---------------------##
###########################################################
if __name__ == '__main__':
    print("---------- Starting Worker-Agent ----------")

    print("$ Loading configuration ...")
    amqp_cfg = amqp_config.get_conf()
    auth_cfg = amqp_cfg["auth"]
    local_cfg = amqp_cfg["local"]
    master_cfg = amqp_cfg["master"]
    try:
        local_connection = Connection(
            hostname=local_cfg['hostname'],
            port=local_cfg['port'],
            userid=auth_cfg['username'],
            password=auth_cfg['password'],
            virtual_host=auth_cfg['vhost'],
        )
    except Exception as e:
        print("Error connecting to local broker.")
        raise e

    print("$ Activating Agent ...")
    wa = WorkerAgent(local_connection)

    print("--> Activating workerside taskmanager...")
    wa.activate_task_manager()
    resources = wa.get_total_resources()
    print("    Resources: %s"%(resources,))

    print("--> Establish connection to master node...")
    master_url = "amqp://" + \
        auth_cfg["username"] + ":" + auth_cfg["password"] + "@" + \
        master_cfg["hostname"] + ":" + str(master_cfg["port"]) + "/" + \
        auth_cfg["vhost"]
    master_cfg["master_url"] = master_url
    print("    Master url: %s"%(master_url,))
    wa.establish_master_conn(master_cfg)

    print("--> Registering worker to master ...")
    node_id = local_cfg['node_id']
    self_amqp_url ='amqp://'\
        + auth_cfg['username'] + ':' + auth_cfg['password']\
        + '@'\
        + local_cfg['ipaddr'] + ':' + str(local_cfg['port']) + "/"\
        + auth_cfg['vhost']
    reg_args = {
        'name': node_id,
        'amqp_url': self_amqp_url,
        'total_cpu': resources['total_cpu'],
        'total_gpu': resources['total_gpu']
    }
    wa.publish_mesg_register_node(reg_args)

    print("$ Start running ...")
    wa.run()
