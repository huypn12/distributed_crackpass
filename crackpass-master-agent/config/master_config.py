import sys
import json

class MasterConfig(object):
    """ Configuration store.
    """
    def __init__(self, env):
        self.base_dir = "./config"
        self.env_list = ['dev', 'test', 'product']
        if env not in self.env_list:
            print("Error occured: environment %s is not supported."%env)
            sys.exit(1)
        else:
            self.base_dir += '/' + env


    def load_db_cfg(self, ):
        """ Get proper database config for specified environment
        """
        conf_dir = self.base_dir + '/db_cfg.json'
        cfg = {}
        with open(conf_dir) as db_conf:
            cfg_str = db_conf.read()
            cfg = json.loads(cfg_str)
        return cfg


    def load_const_cfg(self, ):
        """ Load predefined constants
        """
        conf_dir = self.base_dir + '/const_cfg.json'
        cfg = {}
        with open(conf_dir) as const_conf:
            cfg_str = const_conf.read()
            cfg = json.loads(cfg_str)
        return cfg


    def load_amqp_cfg(self, ):
        """ Load ampq connection conf
        """
        conf_dir = self.base_dir + '/amqp_cfg.json'
        cfg = {}
        with open(conf_dir) as amqp_conf:
            cfg_str = amqp_conf.read()
            cfg = json.loads(cfg_str)
        return cfg
