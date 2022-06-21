import copy 
import numpy as np 
import time
from collections import Counter, defaultdict 

from pyCydonia.profiler.PercentileStats import PercentileStats

""" BlockWorkloadStats
    ------------------
    This class generates the read and write statistics related to 
    frequency, IO size, sequentiality, alignment and more.  

    Parameters
    ----------
    lba_size : int 
        size of a logical block address in bytes (Default: 512)
    page_size : int 
        size of a page in bytes (Default: 4096)
"""

class BlockWorkloadStats:

    def __init__(self, lba_size=512, page_size=4096):
        self._lba_size = lba_size 
        self._page_size = page_size 

        self._read_block_req_count = 0 
        self._read_page_access_count = 0 
        self._read_io_request_size_sum = 0
        self._read_seq_count = 0 
        self._read_misalignment_sum = 0 
        self._min_read_page = 0 
        self._max_read_page = 0
        self._read_page_access_counter = Counter()

        self._write_block_req_count = 0 
        self._write_page_access_count = 0 
        self._write_io_request_size_sum = 0 
        self._write_seq_count = 0 
        self._write_misalignment_sum = 0 
        self._min_write_page = 0 
        self._max_write_page = 0
        self._write_page_access_counter = Counter()

        self._read_size_stats = PercentileStats()
        self._write_size_stats = PercentileStats()
        self._jump_distance_stats = PercentileStats()
        self._scan_stats = PercentileStats()

        # tracking information from previous window 
        self._prev_req = None 
        self._prev_window_read_page_access_counter = None 
        self._prev_window_write_page_access_counter = None 
        self._prev_window_read_popularity_map = None 
        self._prev_window_write_popularity_map = None 
        self._scan_length = 0 

        self.start_time = time.time()


    def block_req_count(self):
        """ This functions returns the number of block requests. """
        return self._read_block_req_count + self._write_block_req_count


    def write_block_req_split(self):
        """ This function returns the fraction of block requests that were writes. """
        return self._write_block_req_count/self.block_req_count()


    def io_request_size_sum(self):
        """ This function returns the total IO in bytes. """
        return self._read_io_request_size_sum + self._write_io_request_size_sum 


    def write_io_request_size_split(self):
        """ This function returns the fraction of IO that was for write requests. """
        return self._write_io_request_size_sum/self.io_request_size_sum()


    def page_access_count(self):
        """ This function returns the number of pages accessed. """
        return self._read_page_access_count + self._write_page_access_count


    def write_page_access_split(self):
        """ This function returns the fraction of write page requests. """
        return self._write_page_access_count/self.page_access_count()

    
    def seq_count(self):
        """ This functions returns the number of sequential block accesses. """
        return self._read_seq_count + self._write_seq_count


    def write_seq_split(self):
        """ This function returns the number of sequential block accesses that were writes. """
        return self._write_seq_count/self.seq_count()


    def range(self):
        """ This functions returns the byte range accessed in a workload. """
        return self._page_size * (max(self._max_read_page, self._max_write_page) \
                    - min(self._min_read_page, self._min_write_page))


    def read_range(self):
        """ This functions returns the byte range read in a workload. """
        return self._page_size * (self._max_read_page - self._min_read_page)


    def write_range(self):
        """ This functions returns the byte range read in a workload. """
        return self._page_size * (self._max_write_page - self._min_write_page)
    

    def misalignment_sum(self):
        """ This functions returns the total byte misalignment in block requests. """
        return self._read_misalignment_sum + self._write_misalignment_sum
    

    def req_start_offset(self, req):
        """ This function returns the start offset of a block request. """
        return req["lba"] * self._lba_size


    def req_end_offset(self, req):
        """ This function returns the end offset of a block request. """
        return self.req_start_offset(req) + req["size"]

    
    def page_working_set_size(self):
        """ This function returns the size of the working set size. """
        read_wss = set(self._read_page_access_counter.keys())
        write_wss = set(self._write_page_access_counter.keys())
        return self._page_size * len(read_wss.union(write_wss))


    def read_page_working_set_size(self):
        """ This function returns the size of the read working set size. """
        return self._page_size * len(self._read_page_access_counter.keys())


    def write_page_working_set_size(self):
        """ This function returns the size of the write working set size. """
        return self._page_size * len(self._write_page_access_counter.keys())

    
    def write_page_working_set_size_split(self):
        """ This function returns the fraction of working set size that was written upon. """
        return self.write_page_working_set_size()/self.page_working_set_size()


    def page_popularity_map(self, counter):
        """ This functions returns the map of each page and its popularity. """
        popularity_map = defaultdict(float)
        for key in counter:
            popularity_map[key] = counter[key]/self.page_access_count()
        return popularity_map


    def _track_req_alignment(self, req):
        """ This function tracks the byte alignment of block requests. 

            Parameters
            ----------
            req : object 
                an object containing block request features
        """

        if req["op"] == "r":
            self._read_misalignment_sum += req["front_misalign"]
            self._read_misalignment_sum += req["rear_misalign"]
        elif req["op"] == "w":
            self._write_misalignment_sum += req["front_misalign"]
            self._write_misalignment_sum += req["rear_misalign"]


    def _track_seq_access(self, req):
        """ A sequential block request starts at the same 
            offset where the previous block request ended. 

            The operation type (read/write) of a sequential 
            access is determined by the following request
            and the operation of the previous request is 
            irrelevant. 

            Parameters
            ----------
            req : object 
                an object containing block request features
        """

        if self._prev_req == None:
            return 

        """ We are comparing the start offset of current request 
            to the end offset of previous request. """
        start_offset = self.req_start_offset(req)
        prev_end_offset = self.req_end_offset(self._prev_req)
        self._jump_distance_stats.add_data(start_offset-prev_end_offset)
        if start_offset == prev_end_offset:
            # Sequential! 
            if req["op"] == 'r':
                self._read_seq_count += 1 
            else:
                self._write_seq_count += 1
        

    def _track_op_type(self, req):
        """ This function tracks the request counts for read and write. 

            Parameters
            ----------
            req : object 
                an object containing block request features
        """

        if req["op"] ==  'r':
            self._read_block_req_count += 1
            self._read_io_request_size_sum = req["size"]
            self._read_page_access_count += req["end_page"] - req["start_page"] + 1
            self._min_read_page = max(self._min_read_page, req["start_page"])
            self._max_read_page = max(self._max_read_page, req["end_page"])
            self._read_size_stats.add_data(req["size"])
        elif req["op"] == 'w':
            self._write_block_req_count += 1 
            self._write_io_request_size_sum = req["size"]
            self._write_page_access_count += req["end_page"] - req["start_page"] + 1
            self._min_write_page = max(self._min_write_page, req["start_page"])
            self._max_write_page = max(self._max_write_page, req["end_page"])
            self._write_size_stats.add_data(req["size"])
        else:
            raise ValueError("Operation {} not supported. Only 'r' or 'w'".format(req["op"]))


    def _get_popularity_data(self, counter, prev_counter, prev_map):
        """ Get the popularity counter 

            Parameters
            ----------
            counter : Counter 
                the counter values of the current window 
            prev_counter : Counter 
                the counter values at the previous window 
            prev_map : defauldict 
                the map of item popularity of the previous window 

            Return 
            ------
            popularity_change_array : np.array 
                numpy array containing popularity change of each item 
            popularity_map : defaultdict 
                map of item to popularity 
        """

        if prev_counter is None:
            return None, None 

        delta = counter - prev_counter
        popularity_map = self.page_popularity_map(delta)
        
        if prev_map is not None:
            prev_window_key_set = set(prev_map.keys())
            cur_window_key_set = set(popularity_map)
            final_key_set = set.union(cur_window_key_set, prev_window_key_set)

            popularity_change_array = PercentileStats(size=len(final_key_set))
            for key in final_key_set:
                popularity_change_array.add_data(popularity_map[key] - prev_map[key])
            return popularity_change_array, popularity_map
        else:
            return None, popularity_map
        

    def _track_popularity(self, req):
        """ Change in popularity of an item. 

            Parameters
            ----------
            req : object 
                an object containing block request features
        """

        if req['op'] == 'r': 
            for page_index in range(req["start_page"], req["end_page"]+1):
                if page_index not in self._read_page_access_counter:
                    self._scan_length += 1 
                else:
                    if self._scan_length > 0:
                        self._scan_stats.add_data(self._scan_length)
                        self._scan_length = 0 
                self._read_page_access_counter[page_index] += 1
        else:
            for page_index in range(req["start_page"], req["end_page"]+1):
                self._write_page_access_counter[page_index] += 1
                self._scan_length += 1


    def add_request(self, block_req):
        """ Update the statistics based on a block request 
            provided by the user. 

            Parameters
            ----------
            block_req : object 
                an object containing block request features
        """

        self._track_op_type(block_req)
        self._track_seq_access(block_req)
        self._track_req_alignment(block_req)
        self._track_popularity(block_req)
        self._prev_req = block_req

    
    def _snap_req_stats(self):
        """ This function returns an array of statistics 
            of block requests in string format which can 
            be printed or written to file. 
        """

        return [str(_) for _ in [self.block_req_count(), 
                                    self._read_block_req_count,
                                    self._write_block_req_count,
                                    self.write_block_req_split()]]


    def _snap_page_stats(self):
        """ This function returns an array of statistics 
            of page requests in string format which can 
            be printed or written to file. 
        """

        return [str(_) for _ in [self.page_access_count(), 
                                    self._read_page_access_count,
                                    self._write_page_access_count,
                                    self.write_page_access_split()]]


    def _snap_io_stats(self):
        """ This function returns an array of statistics 
            of read and write IO sizes in string format which can 
            be printed or written to file. 
        """

        return [str(_) for _ in [self.io_request_size_sum(), 
                                    self._read_io_request_size_sum,
                                    self._write_io_request_size_sum,
                                    self.write_io_request_size_split()]]

    
    def _snap_seq_stats(self):
        """ This function returns an array of statistics 
            of related of sequentiality in string format which can 
            be printed or written to file. 
        """

        return [str(_) for _ in [self.seq_count(), 
                                    self._read_seq_count,
                                    self._write_seq_count,
                                    self.write_seq_split()]]

    
    def _snap_alignment_stats(self):
        """ This function returns an array of statistics 
            of related to byte alignment of block requests
            in string format which can be printed or written to file. 
        """

        return [str(_) for _ in [self.misalignment_sum(), 
                                    self._read_misalignment_sum,
                                    self._write_misalignment_sum]]


    def _snap_range_stats(self):
        """ This function returns an array of statistics 
            of related to range of accesses of block requests
            in string format which can be printed or written to file. 
        """

        return [str(_) for _ in [self.range(), 
                                    self._max_read_page - self._min_read_page,
                                    self._max_write_page - self._min_write_page]]

                    
    def _snap_working_set_size(self):
        """ This function returns an array of working set size statistics
            in string format which can be printed or written to file. 
        """
        
        return [str(_) for _ in [self.page_working_set_size(),
                                    self.read_page_working_set_size(),
                                    self.write_page_working_set_size(),
                                    self.write_page_working_set_size_split()]]


    def _snap_popularity_stat(self):
        """ This function returns an array of popularity statistics 
            of each page in string format which can be printed or written 
            to file. 
        """

        r_delta_array, r_popularity_map = self._get_popularity_data(self._read_page_access_counter,
                                                                    self._prev_window_read_page_access_counter,
                                                                    self._prev_window_read_popularity_map)
        w_delta_array, w_popularity_map = self._get_popularity_data(self._write_page_access_counter,
                                                                    self._prev_window_write_page_access_counter,
                                                                    self._prev_window_write_popularity_map)

        self._prev_window_read_page_access_counter = copy.deepcopy(self._read_page_access_counter)
        self._prev_window_write_page_access_counter = copy.deepcopy(self._write_page_access_counter)

        self._prev_window_read_popularity_map = r_popularity_map
        self._prev_window_write_popularity_map = w_popularity_map

        out_str_array = []
        if r_delta_array is not None:
            out_str_array += r_delta_array.get_row()
        
        if w_delta_array is not None:
            out_str_array += w_delta_array.get_row()

        return out_str_array


    def snapshot(self, window_index, ts, out_handle=None, force_print=False):
        """ This function outputs the features of a block workload. 

            Parameters
            ----------
            window_index : int 
                the index of time window 
            ts : int 
                the timestamp 
            out_handle : int (Optional)
                the handle to write to (Default: None)
            force_print : bool (Optional)
                whether to write to file and print as well (Default: False)
        """

        snap_data = []
        snap_data += self._snap_req_stats()
        snap_data += self._snap_page_stats()
        snap_data += self._snap_io_stats()
        snap_data += self._snap_seq_stats()
        snap_data += self._snap_alignment_stats()
        snap_data += self._snap_range_stats()
        snap_data += self._snap_working_set_size()
        snap_data += self._snap_popularity_stat()
        snap_data += self._read_size_stats.get_row()
        snap_data += self._write_size_stats.get_row()
        snap_data += self._jump_distance_stats.get_row()
        snap_data += self._scan_stats.get_row()

        self._read_size_stats = PercentileStats()
        self._write_size_stats = PercentileStats()
        self._jump_distance_stats = PercentileStats()
        self._scan_stats = PercentileStats()

        if window_index % 10 == 0:
            cur_time = time.time()
            print("Window: {}, TS: {}, Elasped: {}".format(window_index, 
                                                            ts/1e6,
                                                            (cur_time-self.start_time)/60))

        out_str = ",".join(snap_data)
        if out_handle is not None:
            out_handle.write("{},{},{}\n".format(window_index, ts, out_str))
            if force_print:
                print(out_str)
        else:
            print(out_str)