from model import CrackpassModel


class NodeModel(CrackpassModel):

    ################################################################
    ##------------------------- Nodes ----------------------------##
    ################################################################
    def insert_node(self, args):
        """ Insert a normal user into db
        @param user
        @return bool
        """
        conn = self.get_conn()
        cursor = self.get_cursor(conn)
        try:
            # Insert into auth_user table;
            query_str = """
            INSERT INTO node(
            node_name,
            amqp_url,
            total_cpu,
            total_gpu,
            avail_cpu,
            avail_gpu,
            current_state
            ) VALUES(%s,%s,%s,%s,%s,%s,%s)
            """
            data = (
                args['name'],\
                args['amqp_url'],\
                args['total_cpu'],\
                args['total_gpu'],\
                args['total_cpu'],\
                args['total_gpu'],\
                'ONLINE'
            )
            cursor.execute(query_str, data)
            conn.commit()
            inserted_id = cursor.lastrowid
            return inserted_id
        except Exception as e:
            self.logger.error("[ insert_node ] Query=%s ; Exception=%s"%(
                "", str(e),))
            return -1
        finally:
            cursor.close()
            conn.close()


    ##@huypn
    def update_node_avail_resources(self, args):
        """ Update available resources
        """
        conn = self.get_conn()
        cursor = self.get_cursor(conn)
        try:
            query_str = """
            UPDATE node
            SET avail_cpu=%s,avail_gpu=%s
            WHERE node_id=%s
            """
            data = (
                args['avail_cpu'],
                args['avail_gpu'],
                args['node_id'],
            )
            cursor.execute(query_str, data)
            conn.commit()
        except Exception as e:
            self.logger.error("[ update_node_avail_resources ] Query=%s, Exception=%s",\
                              cursor._last_executed, str(e))
            return -1
        finally:
            cursor.close()
            conn.close()


    ##@huypn
    def update_node_state(self, args):
        """ Update node state: ONLINE or OFFLINE
        """
        conn = self.get_conn()
        cursor = self.get_cursor(conn)
        try:
            query_str = """
            UPDATE node
            SET avail_cpu=%s,avail_gpu=%s
            WHERE node_id=%s
            """
            data = (
                args['new_state'],
                args['node_id']
            )
            cursor.execute(query_str, data)
            conn.commit()
        except Exception as e:
            self.logger.error("[ update_node_avail_resources ] Query=%s, Exception=%s",\
                              cursor._last_executed, str(e))
            return -1
        finally:
            cursor.close()
            conn.close()


    ##@huypn
    def get_node_id_by_name(self, args):
        """Check if a node is available on db
        - if yes -> return node_id
        - if no  -> return -1
        """
        conn = self.get_conn()
        cursor = self.get_cursor(conn)
        try:
            query_str = """
            SELECT node_id
            FROM node
            WHERE node.name=%s
            """
            data = (args['name'],)
            cursor.execute(query_str, data)
            conn.commit()
            rows = cursor.fetchall()
            if len(rows) > 0:
                return rows[0]['node_id']
            else:
                return -1
        except Exception as e:
            self.logger.error("[ get_node_id_by_name ] Query=%s; Exception=%s",
                              cursor._last_executed, str(e))
            return -1
        finally:
            cursor.close()
            conn.close()


    ##@huypn:
    def get_all_nodes(self, ):
        """
        """
        conn = self.get_conn()
        cursor = self.get_cursor(conn)
        try:
            query_str = """
            SELECT *
            FROM node
            ORDER BY node_id ASC
            """
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


    ##@huypn:
    def get_allocatable_nodes(self, ):
        """ Get nodes with allocatable resources """
        conn = self.get_conn()
        cursor = self.get_cursor(conn)
        try:
            query_str = """
            SELECT *
            FROM node
            WHERE avail_cpu > 0 OR avail_gpu > 0
            ORDER BY avail_cpu
            """
            cursor.execute(query_str)
            conn.commit()
            rows = cursor.fetchall()
            return rows
        except Exception as e:
            self.logger.error("[ get_allocatable_nodes ] Exception :%s"%(
                 str(e)
            ))
        finally:
            cursor.close()
            conn.close()

