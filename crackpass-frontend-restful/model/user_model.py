import hashlib

from crackpass_model import CrackpassModel


class UserModel(CrackpassModel):

    def insert_user(self, user):
        """ Insert a normal user into db
        @param: user
        @return: bool
        """
        conn = self.get_conn()
        cursor = self.get_cursor(conn)
        try:
            # Insert into auth_user table;
            query_str = """
            INSERT INTO auth_user(
            username, password, firstname, lastname, joined_date, role_name, current_state
            ) VALUES(%s, %s, %s, %s, NOW(), %s, %s)
            """
            data = (user['username'],\
                    hashlib.sha256(user['password']).hexdigest(),\
                    user['firstname'],\
                    user['lastname'],\
                    'USER',
                    'ACTIVE')
            cursor.execute(query_str, data)
            conn.commit()
            inserted_id = cursor.lastrowid
            # Return newuser's id
            return inserted_id
        except Exception as e:
            self.logger.error("[ insert_user ] Query=%s ; Exception=%s"%(cursor._last_executed, str(e)))
            return -1
        finally:
            cursor.close()
            conn.close()


    ##@huypn: ADMIN
    def get_all_users(self, args):
        """ ADMIN. Get all user info.
        """
        conn = self.get_conn()
        cursor = self.get_cursor(conn)
        try:
            query_str = """
            SELECT username, firstname, lastname, role, current_state
            FROM auth_user
            """
            cursor.execute(query_str)
            conn.commit()
            rows = cursor.fetchall()
            return rows
        except Exception as e:
            self.logger.error("[ get_all_users ] Query=%s ; Exception=%s"%(cursor._last_executed, str(e)))
            return -1
        finally:
            cursor.close()
            conn.close()


    ##@huypn:
    def get_user_info(self, args):
        conn = self.get_conn()
        cursor = self.get_cursor(conn)
        try:
            query_str = """
            SELECT username, firstname, lastname, role, current_state
            FROM auth_user
            WHERE auth_user.user_id=%s
            """
            print(args)
            data = (args['user_id'],)
            cursor.execute(query_str, data)
            conn.commit()
            #@huypn: replace fetchall by fetchone
            # rows = cursor.fetchall()
            user = cursor.fetchone()
            return user
        except Exception as e:
            self.logger.error("[ get_user_info ] Query=%s ; Exception=%s"%(cursor._last_executed, str(e)))
            return -1
        finally:
            cursor.close()
            conn.close()


    ##@huypn
    def update_user_password(self, args):
        """ Update user password; user-privillege only.
        """
        conn = self.get_conn()
        cursor = self.get_cursor(conn)
        try:
            query_str = """
            UPDATE auth_user
            SET password=%s
            WHERE auth_user.user_id=%s
            """
            hashed_password = hashlib.sha256(args['new_password']).hexdigest()
            data = (hashed_password, args['user_id'], )
            cursor.execute(query_str, data)
            conn.commit()
        except Exception as e:
            self.logger.error("[ update_task_progress ]Query=%s ; Exception=%s"%(cursor._last_executed, str(e)))
            return -1
        finally:
            cursor.close()
            conn.close()

    ##@huypn
    def verify_user_password(self, args):
        """ Verify user password """
        conn = self.get_conn()
        cursor = self.get_cursor(conn)
        try:
            query_str = """
            SELECT user_id, role, username, firstname, lastname
            FROM auth_user
            WHERE auth_user.username=%s AND auth_user.password=%s
            """
            hashed_password = hashlib.sha256(args['password']).hexdigest()
            data = (args['username'], hashed_password, )
            cursor.execute(query_str, data)
            conn.commit()
            row = cursor.fetchone()
            return row
        except Exception as e:
            self.logger.error("[ update_task_progress ]Query=%s ; Exception=%s"%(cursor._last_executed, str(e)))
            return -1
        finally:
            cursor.close()
            conn.close()

