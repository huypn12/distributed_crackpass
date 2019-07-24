

class MD5WorkerTask(object):
    # Worker task
    # Stores actual worker instance pool
    def __init__(self, controller, params):
        self.controller = controller
        self.task_id = params["task_id"]

        pass

    def push_request(self, args):
        task_id = args['task_id']
        task_subid = args['task_subid']     # Pool number: cpu or gpu

        pass
