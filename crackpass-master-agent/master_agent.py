import json
import logging

from kombu import Connection, Queue, Exchange
from kombu.mixins import ConsumerMixin
from kombu.pools import producers

from config import amqp_config
from master_task_manager import MasterTaskManager


class MasterAgent(ConsumerMixin):
    """Master-agent main running daemon.
    """
    def __init__(self, connection):
        """ Constructor """
        # Set logger
        logging.basicConfig(filename='log/master-agent.log', level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        # Establish master self-connect
        self.connection = connection
        self.exchange = Exchange('e_master', 'direct', durable=True)
        self.q_register = Queue('q_register',
                                exchange=self.exchange,
                                routing_key='rk_register')
        self.q_request = Queue('q_request',
                               exchange=self.exchange,
                               routing_key='rk_request')
        self.q_finished = Queue('q_finished',
                                exchange=self.exchange,
                                routing_key='rk_finished')
        # Create task manager & database handler
        self.task_manager = MasterTaskManager(self)
        # Store publisher to worker
        self.workers = []
        # worker: { 'node_id': xx, 'node_amqp_link':  xxx, 'exchange': xxxx}
        self.name = ''


    ###########################################################
    ##------------------ Mesg Listening ---------------------##
    ###########################################################
    def on_mesg_register(self, body, mesg):
        """ Callback function, serving q_register
        """
        # Verbose
        self.logger.info("< Q_REGISTER > delivered body=%s;mesg=%s;"\
                         %(body,mesg,))
        try:
            args = body
            node_id = self.task_manager.insert_node(args)
            args['node_id'] = node_id
            self.workers.append(args)
        except Exception as e:
            self.logger.error("[ on_mesg_register ] Error occurred: %s"%\
                              (str(e),))
        finally:
            mesg.ack()


    def on_mesg_request(self, body, mesg):
        """ Callback function, serving q_request
        """
        self.logger.info("< Q_REQUEST > delivered body=%s;mesg=%s"%(body,mesg,))
        try:
            args = body
            # Get the request from proper task
            data = self.task_manager.request_data_from_task(args)
            # Send back response data
            if data is None:
                return
            else:
                # self.publish_mesg_response(args['name'], data)
                data['task_local_id'] = args['task_local_id']
                self.publish_mesg_response(args['amqp_url'], data)
        except Exception as e:
            self.logger.error("[ on_mesg_request ] Error occurred: %s"%\
                              (str(e),))
        finally:
            mesg.ack()


    def on_mesg_push_result(self, body, mesg):
        self.logger.info("<Q_RESULT> delivered body=%s;mesg=%s"%(body,mesg))
        try:
            args = body
            self.task_manager.push_result_to_task(args)
        except Exception as e:
            self.logger.error("[ on_mesg_push_result ] Error occurred: %s"%\
                              (str(e),))
        finally:
            mesg.ack()


    def on_mesg_finished(self, body, mesg):
        """ Callback function, serving q_finished
        """
        self.logger.info("< Q_FINISHED > delivered body=%s;mesg=%s"%(body,mesg,))
        try:
            args = body
            self.task_manager.update_completed_task(args)
        except Exception as e:
            self.logger.error("[ on_mesg_finished ] Error occurred: %s"%\
                              (str(e),))
        finally:
            mesg.ack()


    def get_consumers(self, Consumer, channel):
        return [
            Consumer(queues=[self.q_register],
                     callbacks=[self.on_mesg_register],
                     accept=['json']),
            Consumer(queues=[self.q_request],
                     callbacks=[self.on_mesg_request],
                     accept=['json']),
            Consumer(queues=[self.q_finished],
                     callbacks=[self.on_mesg_finished],
                     accept=['json'],)
        ]


    ###########################################################
    ##---------------- Mesg Publishing  ---------------------##
    ###########################################################

    def publish_mesg_create_task(self, amqp_url, args):
        """ Send message to q_create_task of node_id """
        self.logger.info("<Publisher> publish create task %s <to> %s"%(args, amqp_url,))
        try:
            #@huypn: remove dict searching
            # remote_amqp_link = node['node_amqp_url']
            # remote_amqp_conn = Connection(remote_amqp_link)
            remote_amqp_conn = Connection(amqp_url)
            with producers[remote_amqp_conn].acquire(block=True) as producer:
                ex = Exchange("e_worker", type="direct")
                rk = 'rk_create_task'
                producer.publish(
                    args,
                    exchange=ex,
                    routing_key=rk,
                    serializer='json'
                )
        except Exception as e:
            self.logger.error("publish_remote: create_task %s"%(str(e),))
            return -1


    def publish_mesg_kill(self, amqp_url, args):
        """ Send message to q_kill_task of node_id """
        self.logger.info("<Publisher> publish kill task %s <to> %s"%(args, amqp_url,))
        try:
            remote_amqp_conn = Connection(amqp_url)
            with producers[remote_amqp_conn].acquire(block=True) as producer:
                ex = Exchange("e_worker", type="direct")
                rk = 'rk_kill_task'
                producer.publish(
                    args,
                    exchange=ex,
                    routing_key=rk,
                    serializer='json'
                )
        except Exception as e:
            self.logger.error("publish_remote: kill_task %s"%(str(e),))
            return -1


    def publish_mesg_response(self, amqp_url, args):
        """ Send message to q_response_data of node_id """
        self.logger.info("<Publisher> publish response %s <to> %s"%(args, amqp_url,))
        try:
            remote_amqp_conn = Connection(amqp_url)
            with producers[remote_amqp_conn].acquire(block=True) as producer:
                ex = Exchange("e_worker", type="direct")
                rk = 'rk_response_data'
                producer.publish(
                    args,
                    exchange=ex,
                    routing_key=rk,
                    serializer='json'
                )
        except Exception as e:
            self.logger.error("publish_remote: %s"%(str(e),))
            return -1




###########################################################
##---------------- Mesg Publishing  ---------------------##
###########################################################
import sys


if __name__ == '__main__':
    print("------ CrackPass MasterAgent ------")

    # Load configuration
    print("$ Loading config files ...")
    env = 'dev'
    amqp_cfg = amqp_config.get_conf()
    auth_cfg = amqp_cfg["auth"]
    local_cfg = amqp_cfg["local"]
    ma = None
    try:
        print("--> Loaded configuration %s"%(str(amqp_cfg),))
        local_connection = Connection(
            hostname=local_cfg['hostname'],
            port=local_cfg['port'],
            userid=auth_cfg['username'],
            password=auth_cfg['password'],
            virtual_host=auth_cfg['vhost']
        )
        # Create new agent
        print("$ Activating agent ...")
        ma = MasterAgent(local_connection)
    except Exception as e:
        print(str(e))
        print("Error connecting to the broker...")
        sys.exit()

    print("$ Launching task manager...")
    ma.task_manager.launch()

    # Activate created agent.
    print("$ Running daemon...")
    ma.run()
