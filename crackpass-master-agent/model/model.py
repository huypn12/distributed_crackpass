import logging
import MySQLdb

try:
    from config import db_config
except Exception as e:
    raise e

class CrackpassModel(object):
    def __init__(self,):
        #
        dbconfig = db_config.get_conf()
        self.db_hostname = dbconfig["hostname"]
        self.db_database = dbconfig["database"]
        self.db_username = dbconfig["username"]
        self.db_password = dbconfig["password"]
        #
        logging.basicConfig(filename='/tmp/crackpass_backend_masteragent_model.log', level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)


    def get_conn(self,):
        conn = MySQLdb.connect(self.db_hostname,
                               self.db_username,
                               self.db_password,
                               self.db_database)
        return conn


    def get_cursor(self, conn):
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)
        return cursor
