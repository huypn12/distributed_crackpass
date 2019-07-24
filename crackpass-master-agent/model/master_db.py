import logging
import json
import MySQLdb as mdb

from master_config import MasterConfig


##@huypn
# -->   2015-10-09T14:39:11 Changed name from MasterAgentDB to MasterBackendDB
#       Reason: to distiguish db accessible methods of backend and frontend.
####
#class MasterAgentDB(object):
class MasterBackendDB(object):
    """ Database CRUD operations
    -> backend side, no user-management required.
    """

    ##@huypn
    def __init__(self, ):
        # Logging configuration.
        logging.basicConfig(filename='log/master-agent-db.log', level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        # Config loader
        self.config_store = MasterConfig('dev')
        self.cfg = self.config_store.load_db_cfg()
        self.defined_flags = self.config_store.load_const_cfg()
        ## db connect args
        self.db_host = self.cfg['db_host']
        self.db_user = self.cfg['db_user']
        self.db_pass = self.cfg['db_pass']
        self.db_name = self.cfg['db_name']


    def get_conn(self, db):
        return mdb.connect(self.db_host,
                           self.db_user,
                           self.db_pass,
                           db)


    def get_cursor(self, conn):
        return conn.cursor(mdb.cursors.DictCursor)



    ################################################################
    ##------------------------- users ----------------------------##
    ################################################################
    #@huypn: no need of user management on backend

    ################################################################
    ##------------------------- tasks ----------------------------##
    ################################################################

    ##@huypn
    def insert_task(self, args):
        """
        @param: task: dict {'task_id': '', 'params': }
        @return: task_id
        """
        conn = self.get_conn(self.db_name)
        cursor = self.get_cursor(conn)
        try:
            # insert into 'task'; inserted information is enough for queuing
            query_str = """
            INSERT INTO task(
            params,
            type_name,
            n_cpu,
            n_gpu,
            progress,
            eta,
            current_state
            ) VALUES(%s,%s,%s,%s,%s,%s)
            """
            data = (json.dumps(args['params']),
                    args['type_name'],
                    args['n_cpu'],
                    args['n_gpu'],
                    '0/0',
                    '',
                    'IN_QUEUE')
            cursor.execute(query_str, data)
            conn.commit()
            # insert in to relational-description table
            inserted_id = cursor.lastrowid
            query_str = """
            INSERT INTO user_own_task(
            user_id,
            task_id
            ) VALUES(%s,%s)
            """
            data = (args['user_id'], inserted_id, )
            cursor.execute(query_str, data)
            conn.commit()
            return inserted_id
        except Exception as e:
            self.logger.error("[ insert_task ] query=%s ; exception=%s"\
                              %(cursor._last_executed, str(e)))
            return -1


    ##@huypn
    def get_queued_tasks_by_arrived_time(self, ):
        """ query queued task
        @param:
        @return:
        """
        conn = self.get_conn(self.db_name)
        cursor = self.get_cursor(conn)
        try:
            query_str = """
                SELECT *
                FROM task
                WHERE task.current_state = 'IN_QUEUE'
                ORDER BY task.created_date ASC
                """
            cursor.execute(query_str)
            rows = cursor.fetchall()
            return rows
        except Exception as e:
            self.logger.error("[ get_queued_tasks_by_arrived_time ] Query=%s ; Exception=%s"%(cursor._last_executed, str(e)))
            return -1

    ##@huypn
    def get_highest_priority_queued_task(self, ):
        """
        @param
        """
        waiting_tasks = self.get_queued_tasks_by_arrived_time()
        if waiting_tasks is not None and len(waiting_tasks) > 0:
            return waiting_tasks[0]
        else:
            self.logger.error("[ get_highest_priority_queued_task ] Error occured during query.")
            return -1


    ##@huypn
    def get_tasks_by_state(self, state_flag):
        """ Get task by specific flag
        """
        if state_flag not in self.defined_flags:
            self.logger.error("State flag %s is not defined."%state_flag)
            return -1

        conn = self.get_conn(self.db_name)
        cursor = self.get_cursor(conn)
        try:
            query_str = """ SELECT * FROM task WHERE current_state=%s"""
            data = (state_flag,)
            cursor.execute(query_str, data)
            conn.commit()
            rows = cursor.fetchall()
            return rows
        except Exception as e:
            self.logger.error("Query=%s ; exception=%s"%(cursor._last_executed, str(e)))
            return -1


    ##@huypn
    def get_tasks_by_user_id(self, user_id):
        """
        TODO:
        """
        conn = self.get_conn(self.db_name)
        cursor = self.get_cursor(conn)
        try:
            query_str = """
            SELECT *
            FROM task, auth_user, user_own_task
            WHERE auth_user.user_id = user_own_task.user_id
            AND user_own_task.task_id = task.task_id
            ORDER BY created_date
            """
            cursor.execute(query_str)
            conn.commit()
            rows = cursor.fetchall()
            return rows
        except Exception as e:
            self.logger.error("[ get_tasks_by_user_id ] Query=%s ; Exception=%s"%(cursor._last_executed, str(e)))
            return -1


    ##@huypn
    def get_all_tasks_by_user_id(self, ):
        """ Get all tasks, regardless its owner
        NOTE: restricted usage to admin only
        """
        conn = self.get_conn(self.db_name)
        cursor = self.get_cursor(conn)
        try:
            query_str = """
            SELECT task.task_id, task.type_name, task.result, task.parameter,
            DATE_FORMAT(task.created_date, '%Y:%M:%D - %H:%i:%S') as created_date_str,
            user_own_task.user_id
            FROM task, user_own_task
            WHERE task.task_id = user_own_task.task_id
            ORDER BY created_date
            """
            cursor.execute(query_str)
            conn.commit()
            rows = cursor.fetchall()
            return rows
        except Exception as e:
            self.logger.error("[ get_all_task_by_user_id ] Query=%s ; Exception=%s"%(cursor._last_executed, str(e)))
            return -1


    ##@huypn
    def get_all_tasks(self, ):
        """ Get all tasks, regardless its owner
        NOTE: restricted usage to admin only
        """
        conn = self.get_conn(self.db_name)
        cursor = self.get_cursor(conn)
        try:
            query_str = """
            SELECT task.task_id, task.type_name, task.result, task.parameter,
            DATE_FORMAT(task.created_date, '%Y:%M:%D - %H:%i:%s') as created_date_str,
            user_own_task.user_id
            FROM task, user_own_task
            WHERE task.task_id = user_own_task.task_id
            ORDER BY created_date
            """
            cursor.execute(query_str)
            conn.commit()
            rows = cursor.fetchall()
            return rows
        except Exception as e:
            self.logger.error("[ get_all_task ] Query=%s ; Exception=%s"%(cursor._last_executed, str(e)))
            return -1


    ##@huypn
    def update_task_state(self, args):
        """ Update task state only
        @param: one row in ^ return
        @return: bool
        """
        conn = self.get_conn(self.db_name)
        cursor = self.get_cursor(conn)
        try:
            query_str = """
            UPDATE task
            SET task.current_state=%s
            WHERE task.task_id=%s
            """
            data = (
                args['new_state'],
                args['task_id'],
            )
            cursor.execute(query_str, data)
            conn.commit()
        except Exception as e:
            self.logger.error("[ update_task_state ] Query=%s ; Exception=%s"\
                              %(cursor._last_executed, str(e)))
            return -1


    ##@huypn
    def update_task_progress(self, args):
        """ Update task progress
        @param: task list
        """
        conn = self.get_conn(self.db_name)
        cursor = self.get_cursor(conn)
        try:
            query_str = """
            UPDATE task
            SET task.progress=%s,task.eta=%s
            WHERE task.task_id=%s
            """
            data = (
                args['progress'],
                args['eta'],
                args['task_id']
            )
            cursor.execute(query_str, data)
            conn.commit()
        except Exception as e:
            self.logger.error("[ update_task_progress ]Query=%s ; Exception=%s"\
                              %(cursor._last_executed, str(e)))
            return -1


    ##@huypn
    def update_task_result(self, args):
        """ Update all tasks info,
        """
        conn = self.get_conn(self.db_name)
        cursor = self.get_cursor(conn)
        try:
            query_str = """ UPDATE task SET task.progress=%s WHERE task.task_id=%s"""
            data = (
                args['progress'],
                args['task_id'],
            )
            cursor.execute(query_str, data)
            conn.commit()
        except Exception as e:
            self.logger.error("[ update_task_progress ]Query=%s ; Exception=%s"\
                              %(cursor._last_executed, str(e)))
            return -1


    ##@huypn
    def delete_task(self, task_id):
        """ Delete task from db
        """
        conn = self.get_conn(self.db_name)
        cursor = self.get_cursor(conn)
        try:
            query_str = """ DELETE FROM task WHERE task.task_id == %s"""
            cursor.execute(query_str%(task_id,))
            conn.commit()
        except Exception as e:
            self.logger.error("[ update_task_progress ] Query=%s ; Exception=%s"\
                              %(cursor._last_executed, str(e)))
            return -1


####################################
#### UNIT TEST & INITIALIZATION ####
####################################
if __name__ == '__main__':
    ma_db = MasterBackendDB()

    ## Task CRUD TEST
    alphabet_str = "abcdefghijklmnopqrstuvwxyz"
    numeric_str = "0123456789"
    special_str = ""
    task = {
        'type_name': 'MD5',
        'n_cpu': 4,
        'n_gpu': 1,
        'params': {
            "hash_str": "1733e944925ec01bf60ab15bb1780dfb",
            "charset_str": "abcdefghijklmnopqrstuvwxyz",
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


