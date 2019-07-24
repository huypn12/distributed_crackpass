import time
import datetime
import threading

from master_task import MasterTask


##@huypn
class MasterTaskSHA1Crack(MasterTask):
    """
    Master SHA1Crack task;
    """
    def __init__(self, args):
        """ Constructor
        @param args -- fields: hash_str, charset_str, min_base_len, max_base_len
        """
        self.task_id = args['task_id']
        params = args['params']
        self.t_lock = threading.Lock()
        # Original hash string
        self.hash_str = params['hash_str']
        # Charset
        self.charset_str = params['charset_str']
        self.charset_len = len(params['charset_str'])
        # Plaintext length, from min to max
        self.max_base_len = params['max_base_len']
        self.min_base_len = params['min_base_len']
        self.current_base_len = self.min_base_len
        # Initiate input (Plaintext) base
        self.current_base_idx = 0
        tmp_base = self.compute_base(self.current_base_idx,\
                                     self.current_base_len,\
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
        self.result_str = ''
        # Start time, to calculate eta
        self.start_time = datetime.datetime.now()
        self.duration = 0.0


    ##@huypn
    def get_full_progress(self,):
        # Size of processed data
        checked_size = 0
        for i in range(self.min_base_len, self.current_base_len):
            checked_size += self.partial_size[i]
        checked_size += self.current_base_idx
        ratio = 0.0
        eta = 0
        eta_str = ""
        if self.current_state == "FINISHED":
            if self.active == True:
                current_time = datetime.datetime.now()
                diff_time = current_time - self.start_time
                duration = diff_time.total_seconds()
                eta_str = "Duration=" + str(duration)
                self.elapsed = duration
                self.active = False
            else:
                eta_str = "Duration=" + str(self.elapsed)
        else:
            if checked_size == 0:
                eta_str = "N/A"
            else:
                # calculate eta
                current_time = datetime.datetime.now()
                diff_time = current_time - self.start_time
                diff_time_mm = diff_time.total_seconds()
                # ratio = self.space_size * 1.0 / checked_size
                # eta = (self.space_size * 1.0 / checked_size) * diff_time_mm / 3600.0
                speed = checked_size * 1.0 / diff_time_mm
                eta = (self.space_size - checked_size) / speed / 3600.0
                eta_str = "ETA=" + str(eta)
        return {
            'checked_size': str(checked_size),
            'space_size': str(self.space_size),
            'percentage': str(ratio),
            'eta': str(eta_str)
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
            remaining_size = self.partial_size[self.current_base_len] - self.current_base_idx - 1
            response_offset = 0
            response_baselen = 0
            response_basestr = self.current_base_str
            # If all plaintext_str(s) with size=current_base_len are checked.
            if remaining_size <= suggested_offset:
                response_offset = remaining_size
                # Yet reached the max_base_len: reset base_idx & base_len
                if self.current_base_len < self.max_base_len:
                    response_baselen = self.current_base_len
                    self.current_base_len += 1
                    self.current_base_idx = 0
                    tmp_base = self.compute_base(self.current_base_idx,\
                                                    self.current_base_len,\
                                                    self.charset_str)
                    self.current_base_str = tmp_base['base_str']
                    self.current_base_dictIdx = tmp_base['base_dictIdx']
                # Traversed through all the space
                else:
                    response_baselen = self.current_base_len
                    response_offset = remaining_size
                    self.current_base_idx = self.partial_size[self.current_base_len]
                    self.active = False
            else:
                response_baselen = self.current_base_len
                response_offset = suggested_offset
                self.current_base_idx += suggested_offset
            # Compose response
            response = {}
            response['base_str'] = response_basestr
            response['base_str_len'] = response_baselen
            response['offset'] = response_offset
            # Advance current base
            tmp_base = self.compute_base(self.current_base_idx,
                                            self.current_base_len,
                                            self.charset_str)
            self.current_base_str = tmp_base['base_str']
            self.current_base_dictIdx = tmp_base['base_dictIdx']
            # Logging -- tbi; if needed

            # Return
            return response


    ##@huypn
    def push_result(self, args):
        """ Valid plaintext found
        @param -- args: {'result_str': 'aoeuoaeu'}
        """
        with self.t_lock:
            self.active = False
            self.current_state = 'FINISHED'
            self.result_str = args['result_str']


###################################################
##--------------- UNIT TEST ---------------------##
###################################################
if __name__ == '__main__':
    """ 1. Test SHA1 masterside task
    """
    init_args = {
        'task_id': 1,
        'params': {
            'hash_str': '8632a3ef0e07e2a6d9e9b8e232238232',
            'charset_str': 'abcdefghijklmnopqrstuv',
            'min_base_len': 4,
            'max_base_len': 8
        }
    }
    sample_task = MasterTaskSHA1Crack(init_args)
    print("Space size: ")








































