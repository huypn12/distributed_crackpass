import json

from crackpass_model import CrackpassModel


class TaskModel(CrackpassModel):
    def __init__(self,):
        super(TaskModel, self).__init__()

    ##@huypn
    def insert_task(self, args):
        """
        @param: task: dict {'task_id': '', 'params': }
        @return: task_id
        """
        conn = self.get_conn()
        cursor = self.get_cursor(conn)
        try:
            # Insert into 'task'; inserted information is enough for queuing
            query_str = """
            INSERT INTO task(
            params, type_name, n_cpu, n_gpu, progress, eta, current_state
            ) VALUES(%s,%s,%s,%s,%s,%s,%s)
            """
            data = (json.dumps(args['params']),
                    args['type_name'],
                    args['n_cpu'],
                    args['n_gpu'],
                    json.dumps({ "checked": "0/0", "eta": "infinity" }),
                    "infinity",
                    'IN_QUEUE',)
            cursor.execute(query_str, data)
            conn.commit()
            inserted_id = cursor.lastrowid
            return inserted_id
        except Exception as e:
            self.logger.error("[ insert_task ] Exception=%s",(str(e),))
            return -1
        finally:
            cursor.close()
            conn.close()


    ##@huypn
    def get_queued_tasks_by_arrived_time(self, ):
        """ Query queued task
        @param:
        @return:
        """
        conn = self.get_conn()
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
            self.logger.error("[ get_queued_tasks_by_arrived_time ] Query=%s ; Exception=%s"\
                              %(cursor._last_executed, str(e)))
            return -1
        finally:
            cursor.close()
            conn.close()

    ##@huypn
    def get_highest_priority_queued_task(self, ):
        """
        @param
        """
        waiting_tasks = self.get_queued_tasks_by_arrived_time()
        if waiting_tasks != -1\
                and waiting_tasks is not None\
                and len(waiting_tasks) > 0:
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

        conn = self.get_conn()
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
        finally:
            cursor.close()
            conn.close()


    ##@huypn
    def get_tasks_by_user_id(self, user_id):
        """
        TODO:
        """
        conn = self.get_conn()
        cursor = self.get_cursor(conn)
        try:
            query_str = """
                SELECT *
                FROM task, auth_user, user_own_task
                WHERE auth_user.user_id = user_own_task.user_id
                AND user_own_task.task_id = task.task_id
                ORDER BY created_date """
            cursor.execute(query_str)
            conn.commit()
            rows = cursor.fetchall()
            return rows
        except Exception as e:
            self.logger.error("[ get_tasks_by_user_id ] Query=%s ; Exception=%s"%(cursor._last_executed, str(e)))
            return -1
        finally:
            cursor.close()
            conn.close()


    ##@huypn
    def get_all_tasks_by_user_id(self, ):
        """ Get all tasks, regardless its owner
        NOTE: restricted usage to admin only
        """
        conn = self.get_conn()
        cursor = self.get_cursor(conn)
        try:
            query_str = """
            SELECT task.task_id, task.type_name, task.result, task.parameter,
            DATE_FORMAT(task.created_date, '%Y:%M:%D - %T') as created_date_str
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
        finally:
            cursor.close()
            conn.close()


    ##@huypn
    def get_all_tasks(self, ):

        """ Get all tasks, regardless its owner
        NOTE: restricted usage to admin only
        """
        conn = self.get_conn()
        cursor = self.get_cursor(conn)
        try:
            # @ tmh
            # query_str = """
            # SELECT task.task_id, task.type_name, task.result, task.params,
            # DATE_FORMAT(task.created_date, '%Y:%M:%D - %H:%i:%s') as created_date_str
            # FROM task
            # """
            # f = open('/tmp/lol2', 'w+')
            # query_str = """
            query_str = """
            SELECT
            task.task_id,
            task.type_name,
            task.result,
            task.n_cpu,
            task.n_gpu,
            task.params,
            task.eta,
            task.progress,
            task.current_state,
            DATE_FORMAT(task.created_date, '%Y:%M:%D - %T') as created_date_str
            FROM task
            """
            cursor.execute(query_str)
            conn.commit()
            rows = cursor.fetchall()
            return rows
        except Exception as e:
            self.logger.error("[ get_all_task ] Query=%s ; Exception=%s"%(cursor._last_executed, str(e)))
            return -1
        finally:
            cursor.close()
            conn.close()


    ##@huypn
    def update_task_state(self, args):
        """ Update task state only
        @param: one row in ^ return
        @return: bool
        """
        conn = self.get_conn()
        cursor = self.get_cursor(conn)
        try:
            query_str = """
                UPDATE task
                SET task.current_state=%s
                WHERE task.task_id=%s """
            data = (args['new_state'], args['task_id'], )
            cursor.execute(query_str, data)
            conn.commit()
        except Exception as e:
            self.logger.error("[ update_task_state ] Query=%s ; Exception=%s"%(cursor._last_executed, str(e)))
            return -1
        finally:
            cursor.close()
            conn.close()


