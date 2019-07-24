

class WorkerTask(object):
    C_TASK_STATE = {
        'TASK_STATE_READY':         0,
        'TASK_STATE_RUNNING':       1,
        'TASK_STATE_STOPPED':       2,
        'TASK_STATE_WAITING_DATA':  3,
        'TASK_STATE_FOUND_RESULT':  4,
        'TASK_STATE_FINISHED':      5
    }

    def __init__(self, ):
        pass


    def push_request(self, args):
        """
        Push request to task
        """
        pass


    def working_loop(self, ):
        """"""
        pass

    def start(self, args):
        """
        star
        """
        pass

    def stop(self, args):
        """

        """
        pass

