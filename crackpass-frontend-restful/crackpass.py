import os
import json
import logging

from flask import Flask
from flask import request


from model.user_model import UserModel
from model.task_model import TaskModel
from model.node_model import NodeModel


###########################################################
##--------------- Setting up module path ----------------##
###########################################################


###########################################################
##----------- Setting up global attrs -------------------##
###########################################################

node_model = NodeModel()
user_model = UserModel()
task_model = TaskModel()

logger = logging.getLogger(__name__)
logging.basicConfig(filename='log/master-agent-fe.log', level=logging.DEBUG)


###########################################################
##--------------- Not priv requires   -------------------##
###########################################################
app = Flask(__name__)

@app.route('/get_nodes_info', methods=['GET',])
def get_nodes_info():
    """ GET
    get resources information
    """
    try:
        node_list = node_model.get_all_nodes()
        return json.dumps(node_list)
    except Exception as e:
        logger.error("[ get_nodes_info ] Exception thrown %s"%(str(e),))
        return json.dumps({
            'mesg': 'Error',
            'return': '-1'
        })


##@huypn
@app.route('/login', methods=['POST',])
def login():
    """ Verify creds """
    try:
        args = request.get_json()
        matched_user = user_model.verify_user_password(args)
        if matched_user is None or matched_user == -1:
            result = {
                "user_id": "-1",
                "username": "",
                "role": "",
                "mesg": "Login failed."
            }
        else:
            result = {
                'user_id': matched_user['user_id'],
                'role': matched_user['role'],
                'username': matched_user['username'],
                'mesg': 'OK'
            }
        return json.dumps(result)
    except Exception as e:
        logger.error("[ get_nodes_info ] Exception thrown %s"%(str(e),))
        return json.dumps({
            'mesg': 'Error',
            'user_id': '-1',
            'role': 'NA'
        })



###########################################################
##--------------- Serving normal user -------------------##
###########################################################
#@huypn:
@app.route('/user/get_tasks_info', methods=['GET',])
def user_get_tasks_info():
    """
    @param:
    @return: json: all nodes
    """
    logger.info("< request > get_tasks_info")
    try:
        task_list = task_model.get_all_tasks()
        if task_list == -1:
            return json.dumps({
                'mesg': 'Cant get tasks',
                'code': -1
            })
        else:
            return json.dumps(task_list)
    except Exception as e:
        logger.error("[ get_tasks_info ] Exception thrown %s"%(str(e),))
        return json.dumps({
            'mesg': 'Cant get tasks',
            'code': -1
        })


##@huypn
@app.route('/user/create_task', methods=['POST',])
def user_create_task():
    logger.info("< request > create_task ")
    try:
        new_task = request.get_json()
        new_task['current_state'] = 'IN_QUEUE'
        new_task['progress'] = 0
        new_id = task_model.insert_task(new_task)
        if new_id == -1:
            return -1
        return json.dumps({'mesg': "Task created successfully.",
                           'task_id': new_id})
    except Exception as e:
        logger.error("[ user_create_task ] Exception thrown %s"%(str(e),))
        return -1


##@huypn
@app.route('/user/delete_task', methods=['POST',])
def user_delete_task():
    """ Set task's status to 'DEACTIVATED'
    Actually db-level removal is prohibited here,
    since it creates orphanaged task in the clusters.
    """
    logger.info("< request > delete_task  ")
    try:
        args = request.get_json()
        args['new_state'] = 'DEACTIVATED'
        task_model.update_task_state(args)
        return json.dumps({'mesg': 'Task deactivated.'})
    except Exception as e:
        logger.error("[ user_delete_task ] Exception thrown %s"%(str(e),))
        return -1


##@huypn
@app.route('/user/change_passwd', methods=['POST',])
def user_change_passwd():
    """ Change user password.
    Sample request >
    {
        'user_id': 4,
        'old_password': "oauoaeuo",
        'new_password': "+)[}*)+[*+)*]]"
    }
    Sample return --
    {
    }
    """
    logger.info("< request > delete_task  ")
    try:
        passwd_change_request = request.get_json()
        args = {}
        args['user_id'] = passwd_change_request['user_id']
        args['new_password'] = passwd_change_request['new_password']
        ret = user_model.update_user_password(args)
        return json.dumps(ret)
    except Exception as e:
        logger.error("[ delete_task ] Exception thrown %s"%(str(e),))
        return -1


##@huypn
@app.route('/user/get_user_info', methods=['POST',])
def user_get_user_info():
    """ TODO: /user_get_info
    {
        'user_id': 4
    }
    """
    logger.info("< request > get_user_info  ")
    try:
        req = request.get_json()
        info = user_model.get_user_info(req)
        if info is not None:
            info['name'] = info['firstname'] + ', ' + info['lastname']
            return(json.dumps(info))
        else:
            return(json.dumps({
                'mesg': 'Error',
                'code': -1
            }))
    except Exception as e:
        logger.error("[ get_user_task ] Exception thrown %s"%(str(e),))
        return(json.dumps({
            'mesg': 'Error',
            'code': -1
        }))


###########################################################
##--------------- Serving normal user -------------------##
###########################################################

@app.route('/admin/get_all_users', methods=['GET',])
def admin_get_all_users():
    """ get all users, by admin
    """
    logger.info("< request > admin_get_all_users ")
    try:
        args = {}
        users = user_model.get_all_users(args)
        for user in users:
            user['name'] = user['firstname'] + ', ' + user['lastname']
        return json.dumps(users)
    except Exception as e:
        logger.error("[ admin_get_all_users ] Exception thrown %s"%(str(e),))
        return -1


@app.route('/admin/create_user', methods=['POST',])
def admin_create_user():
    """ User signing up, currently conducted by the admin.
    """
    logger.info("< request > admin_create_user  ")
    ret = {}
    ret['mesg'] = 'Failed.'
    ret['user_id'] = '-1'
    try:
        new_user = request.get_json()
        new_id = user_model.insert_user(new_user)
        ret['user_id'] = new_id
    except Exception as e:
        return (str(e))
    return json.dumps(ret)


###########################################################
##--------------- Serving normal user -------------------##
###########################################################
if __name__=='__main__':
    app.run('0.0.0.0')
