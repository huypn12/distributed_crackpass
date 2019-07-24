import logging
import json
import hashlib
import MySQLdb as mdb

from master_config import MasterConfig

class MasterFrontendDB(object):
    """
    Database CRUD operations
    """

    ##@huypn
    def __init__(self, ):
        self.config_store = MasterConfig('dev')
        self.cfg = self.config_store.load_db_cfg()
        self.defined_flags = self.config_store.load_const_cfg()
        ## db connect args
        self.db_host = self.cfg['db_host']
        self.db_user = self.cfg['db_user']
        self.db_pass = self.cfg['db_pass']
        self.db_name = self.cfg['db_name']
        # logging
        logging.basicConfig(filename='log/master-agent-db.log', level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)


    def get_conn(self, db):
        return mdb.connect(self.db_host,
                           self.db_user,
                           self.db_pass,
                           db)


    def get_cursor(self, conn):
        return conn.cursor(mdb.cursors.DictCursor)


    ################################################################
    ##------------------------- Users ----------------------------##
    ################################################################
    ##@huypn:


    ################################################################
    ##------------------------- Users ----------------------------##
    ################################################################
    ##@huypn ADMIN


    ################################################################
    ##------------------------- Tasks ----------------------------##
    ################################################################


####################################
#### UNIT TEST & INITIALIZATION ####
####################################
if __name__ == '__main__':
    env = 'test'
    ma_db = MasterFrontendDB()

    ## Task CRUD TEST
    task = {
        'type_name': 'MD5',
        'n_cpu':4,
        'n_gpu':1,
        'params': {
            "hash_str": "aoaueoaueaoeu",
            "charset_str": "abcdef",
            "min_base_len": 1,
            "max_base_len": 4
        }
    }
    ma_db.insert_task(task)
    print("Task inserted.")

    tasks = ma_db.get_queued_tasks_by_arrived_time()
    print("Queue size: %s"%len(tasks))

    highest_task = ma_db.get_highest_priority_queued_task()
    print("Task to pop: %s"%str(highest_task))


