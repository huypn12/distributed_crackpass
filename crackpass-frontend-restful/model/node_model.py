from crackpass_model import CrackpassModel


class NodeModel(CrackpassModel):
    def get_all_nodes(self, ):
        """
        """
        conn = self.get_conn()
        cursor = self.get_cursor(conn)
        try:
            query_str = """ SELECT * FROM node ORDER BY node_id ASC """
            cursor.execute(query_str)
            conn.commit()
            rows = cursor.fetchall()
            return rows
        except Exception as e:
            self.logger.error("[ get_all_nodes ] Query=%s ; Exception=%s"%(cursor._last_executed, str(e)))
            return -1
        finally:
            cursor.close()
            conn.close()


    def update_task_on_node(self, args):
        """"""
        conn = self.get_conn()
        cursor = self.get_cursor()
        try:
            data = ""
            task_id = args['task_id']
            node_id = args['task_id']
            n_cpu = args['n_cpu']
            n_gpu = args['n_gpu']
            local_ctx = ""
            query_str = """
            INSERT INTO task_on_node( task_id, node_id, n_cpu, n_gpu, local_ctx )
            VALUES (%s,%s,%s,%s,%s)
            """
            data = (task_id, node_id, n_cpu, n_gpu, local_ctx)
            cursor.execute(query_str, data)
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            self.logger.error(e)
            return -1
        finally:
            cursor.close()
            conn.close()


