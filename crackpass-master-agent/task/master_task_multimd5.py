import time
import datetime
import threading

from master_task import MasterTask

##@huypn
class MasterTaskMultiMD5(MasterTask):
    """
    Master MD5Crack task;
    """
    def __init__(self, args):
        """ Constructor
        @param args -- fields: hash_str, charset_str, min_base_len, max_base_len
        """
        self.task_id = args['task_id']
        params = args['params']
        self.t_lock = threading.Lock()
        # Original hash string
        self.hash_list = params['hash_list']
        self.n_hashes = params['n_hashes']
        # Charset
        self.charset_str = params['charset_str']
        self.charset_len = len(params['charset_str'])
        # Plaintext length, from min to max
        self.max_base_len = params['max_base_len']
        self.min_base_len = params['min_base_len']
        self.current_base_len = self.min_base_len
        # Initiate input (Plaintext) base
        self.current_base_idx = 0       # index oinp
        tmp_base = self.compute_base(self.current_base_idx,
                                     self.current_base_len,
                                     self.charset_str)
        self.current_base_str = tmp_base['base_str']
        self.current_base_dictIdx = tmp_base['base_dictIdx']
        # Calculate cardinality
        self.space_size = 0
        self.partial_size = {}
        for base_len in range(self.min_base_len, self.max_base_len+1):
            len_space_size = 1
            for i in range(0, base_len):
                len_space_size *= self.charset_len
            self.partial_size[base_len] = len_space_size
            self.space_size += len_space_size
        # Active flag
        self.active = True # If active is False, then no more request is accepted.
        self.current_state = 'IN_PROGRESS' # args['current_state']
        # Result store
        self.result_list = []
        # Start time, to calculate eta
        self.start_time = datetime.datetime.now()


    ##@huypn
    def get_progress(self, ):
        checked_size = 0
        for i in range(self.min_base_len, self.current_base_len):
            checked_size += self.partial_size[i]
        checked_size += self.current_base_idx
        return (checked_size / self.space_size)

    ##@huypn
    def get_full_progress(self,):
        # Size of processed data
        checked_size = 0
        for i in range(self.min_base_len, self.current_base_len):
            checked_size += self.partial_size[i]
        checked_size += self.current_base_idx
        # calculate eta
        current_time = datetime.datetime.now()
        diff_time = current_time - self.start_time
        diff_time_mm = diff_time.total_seconds()
        if checked_size == 0:
            eta = -1
        else:
            eta = (self.space_size * 1.0 / checked_size) * diff_time_mm / 3600.0
        return {
            'checked_size': str(checked_size),
            'space_size': str(self.space_size),
            'found': str(len(self.result_list)) + '/' + str(self.n_hashes),
            'eta': str(eta)
        }


    ##@huypn
    def compute_base(self, base_idx, base_len, charset_str):
        """ Compute 'base', by its index, length and charset (the dictionary)
        @param base_idx long
        @param base_len int
        @param charset_str string
        @return { 'base_str': 'something', 'base_dictIdx': [] }
        """
        result = {}
        base = [0] * base_len
        counter = base_idx
        charset_len = len(charset_str)
        # Loop initialize
        i = 0
        carry = 0
        a = 0
        while i < base_len:
            a = base[i] + carry + counter%charset_len
            if a > charset_len:
                carry = 1
                a -= charset_len
            else:
                carry = 0
            base[i] = a
            # increase loop-step
            i += 1
            counter = counter / charset_len
        # Calculate base string
        base_str = ''
        for i in range(0, base_len):
            base_str += charset_str[base[i]]
        # Return
        result['base_str'] = base_str
        result['base_dictIdx'] = base
        return result


    ##@huypn
    def pop_request(self, args):
        """ Pop request
        @param args: { 'offset': (int) }
        @return response: { 'offset': (int), 'base_str': 'aoeuoaeu', 'base_len': (int) }
        """
        # Checking active flag
        if self.active == False:
            return None
        # Threadsafe mutex lock
        print(self.current_base_len)
        with self.t_lock:
            # Get the proposed args
            suggested_offset = args['offset']
            if suggested_offset == 0:
                suggested_offset = 4096
            # remaining_size = self.space_size - self.current_base_idx
            remaining_size = self.partial_size[self.current_base_len] - self.current_base_idx
            response_offset = 0
            # If all plaintext_str(s) with size=current_base_len are checked.
            if remaining_size <= suggested_offset:
                response_offset = remaining_size
                # Yet reached the max_base_len: reset base_idx & base_len
                if self.current_base_len < self.max_base_len:
                    self.current_base_len += 1
                    self.current_base_idx = 0
                    tmp_base = self.compute_base(self.current_base_idx,\
                                                    self.current_base_len,\
                                                    self.charset_str)
                    self.current_base_str = tmp_base['base_str']
                    self.current_base_dictIdx = tmp_base['base_dictIdx']
                # Traversed through all the space
                else:
                    self.current_base_idx = self.partial_size[self.current_base_len]
                    self.active = False
            else:
                response_offset = suggested_offset
                self.current_base_idx += suggested_offset
            # Compose response
            response = {}
            response['base_str'] = self.current_base_str
            response['base_str_len'] = self.current_base_len
            response['offset'] = response_offset
            # Advance current base
            tmp_base = self.compute_base(self.current_base_idx,
                                            self.current_base_len,
                                            self.charset_str)
            self.current_base_str = tmp_base['base_str']
            self.current_base_dictIdx = tmp_base['base_dictIdx']
            # Return
            return response


    ##@huypn
    def push_result(self, args):
        """ Valid plaintext found
        @param -- args: {'result_str': 'aoeuoaeu'}
        """
        with self.t_lock:
            self.result_list.append({
                'hash': args['result_str'],
                'result_str': args['result_str']
            })
            if len(self.result_list) == self.n_hashes:
                self.active = False
                self.current_state = 'COMPLETED'



###################################################
##--------------- UNIT TEST ---------------------##
###################################################
if __name__ == '__main__':
    """ 1. Test MD5 masterside task
    """
    init_args = {
        'task_id': 1,
        'params': {
            'hash_list': [
                "",
                "",
                "",
                ""
            ],
            'n_hashes': 4,
            'charset_str': 'abcdefghijklmnopqrstuv',
            'min_base_len': 4,
            'max_base_len': 8
        }
    }
    sample_task = MasterTaskMultiMD5(init_args)
    while (True):
        response = sample_task.pop_request({'offset': 1228800})
        print(response)




