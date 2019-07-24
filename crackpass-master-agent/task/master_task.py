
##@huypn
class MasterTask(object):
    """ Abstract Master-side task.
    NOTE: requires all args are Dict"""
    def __init__(self, args):
        pass

    ##@huypn @abstract
    def pop_request(self, args):
        """ Abstract method to pop request """
        pass

    def push_result(self, args):
        """ Abstract method to push result """
        pass





