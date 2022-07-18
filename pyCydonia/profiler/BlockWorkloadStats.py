import copy 
import time
import numpy as np 
from collections import Counter, defaultdict 

# logging setup 
# TODO: create a logging class 
import logging
logging.basicConfig(format='%(levelname)s: %(message)s')
logger = logging.getLogger('workload_stat_logger')
logger.setLevel(logging.DEBUG)

from pyCydonia.profiler.PercentileStats import PercentileStats


""" BlockWorkloadStats
    ------------------
    This class generates the read and write statistics related to 
    frequency, IO size, sequentiality, alignment and more.  

    Parameters
    ----------
    lba_size : int (Optional)
        size of a logical block address in bytes (Default: 512)
    page_size : int (Optional)
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

        self._prev_req = None 
        self._scan_length = 0 
        self._start_time = time.time()

        # TODO: clean up the hardcoded values also used in PercentileStats
        self._percentile_step_size = 5 
        self._percentiles_tracked = [1] + list(range(self._percentile_step_size, 101, self._percentile_step_size))

        # call only after percentiles tracked is set
        self.header = self._header()
        self.features = self._feature()


    def block_req_count(self):
        """ This functions returns the number of block requests. """
        return self._read_block_req_count + self._write_block_req_count


    def write_block_req_split(self):
        """ This function returns the fraction of block requests that were writes. """
        return 0 if self.block_req_count()==0 else self._write_block_req_count/self.block_req_count()


    def io_request_size_sum(self):
        """ This function returns the total IO in bytes. """
        return self._read_io_request_size_sum + self._write_io_request_size_sum 


    def write_io_request_size_split(self):
        """ This function returns the fraction of IO that was for write requests. """
        return 0 if self.io_request_size_sum()==0 else self._write_io_request_size_sum/self.io_request_size_sum()


    def page_access_count(self):
        """ This function returns the number of pages accessed. """
        return self._read_page_access_count + self._write_page_access_count


    def write_page_access_split(self):
        """ This function returns the fraction of write page requests. """
        return 0 if self.page_access_count()==0 else self._write_page_access_count/self.page_access_count()

    
    def seq_count(self):
        """ This functions returns the number of sequential block accesses. """
        return self._read_seq_count + self._write_seq_count


    def write_seq_split(self):
        """ This function returns the number of sequential block accesses that were writes. """
        return 0 if self.seq_count()==0 else self._write_seq_count/self.seq_count()


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
        return 0 if self.page_working_set_size()==0 else self.write_page_working_set_size()/self.page_working_set_size()

    
    def read_page_popularity_map(self):
        """ This function returns a map of pages read to its popularity. """
        return self._get_popularity_map(self._read_page_access_counter, self._read_page_access_count)


    def write_page_popularity_map(self):
        """ This function returns a map of pages written to its popularity. """
        return self._get_popularity_map(self._write_page_access_counter, self._write_page_access_count)


    def _get_popularity_map(self, counter, total):
        """ This function returns a defaultdict map of page key to 
            its popularity. 
            
            Parameters
            ----------
            counter : Counter
                the count of access to each page 
            total : int 
                total accesses 
                
            Return 
            ------
            popularity_map : defaultdict(float)
                mapping from page key to popularity (0.0-1.0) """

        popularity_map = defaultdict(float)
        for key in counter.keys():
            popularity_map[key] = counter[key]/total
        return popularity_map


    def _get_popularity_percentile(self, counter, total):
        """ This function returns the percentile stat of page 
            popularity given a counter of access to each 
            page and total page accesses. 

            Parameters
            ----------
            counter : Counter 
                counter of number of accesses to each page 
            total : int 
                total page accesses 

            Return 
            ------
            popularity_stat : PercentileStats 
                PercentileStats object that can generate the percentile """

        popularity_map = self._get_popularity_map(counter, total)
        popularity_stat = PercentileStats(size=len(popularity_map.keys()))
        for page_key in popularity_map:
            popularity_stat.add_data(popularity_map[page_key])
        return popularity_stat
        

    def _get_popularity_change_percentile(self, cur_map, prev_map):
        """ This function returns the percentile of change in 
            page popularity from one window to another. 

            Parameters
            ----------
            cur_map : defaultdict 
                the mapping of page to its access count 
            prev_map : defaultdict 
                the mapping of page to its access count from previous window 
            
            Return 
            ------
            popularity_stat : PercentileStats 
                PercentileStats object that can generate the percentile """

        prev_window_key_set = set(prev_map.keys())
        cur_window_key_set = set(cur_map)
        final_key_set = set.union(cur_window_key_set, prev_window_key_set)
        popularity_stat = PercentileStats(size=len(final_key_set))
        for page_key in final_key_set:
            popularity_stat.add_data(cur_map[page_key]-prev_map[page_key])
        return popularity_stat


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
        size = req["size"]
        start = req["start_page"]
        end = req["end_page"]
        if req["op"] ==  'r':
            self._read_block_req_count += 1
            self._read_io_request_size_sum += size
            self._read_page_access_count += end - start + 1
            self._min_read_page = min(self._min_read_page, start)
            self._max_read_page = max(self._max_read_page, end)
            self._read_size_stats.add_data(size)
        elif req["op"] == 'w':
            self._write_block_req_count += 1 
            self._write_io_request_size_sum += size
            self._write_page_access_count += end - start + 1
            self._min_write_page = min(self._min_write_page, start)
            self._max_write_page = max(self._max_write_page, end)
            self._write_size_stats.add_data(size)
        else:
            raise ValueError("Operation {} not supported. Only 'r' or 'w'".format(req["op"]))


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
            #for page_index in range(req["start_page"], req["end_page"]+1):
             #   self._write_page_access_counter[page_index] += 1
              #  self._scan_length+=1
            start = req["start_page"]
            end = req["end_page"]
            self._write_page_access_counter.update(range(start, end+1))
            self._scan_length += end - start + 1


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
            in string format which can be printed or written 
            to file. 
        """

        read_popularity_stat = self._get_popularity_percentile(self._read_page_access_counter,
                                                                self._read_page_access_count)
        write_popularity_stat = self._get_popularity_percentile(self._write_page_access_counter,
                                                                self._write_page_access_count)

        out_str_array = []
        out_str_array += read_popularity_stat.get_row()
        out_str_array += write_popularity_stat.get_row()

        return out_str_array


    def _snap_popularity_change_stats(self, prev_read_map, prev_write_map):
        """ This function returns an array of percentiles of read and 
            write popularity change. 
            
            Parameters
            ----------
            prev_read_map : defaultdict 
                the mapping of pages read to its access count from previous window 
            prev_write_map : defaultdict 
                the mapping of pages written to its access count from previous window """

        read_popularity_map = self._get_popularity_map(self._read_page_access_counter,
                                                    self._read_page_access_count)
        write_popularity_map = self._get_popularity_map(self._write_page_access_counter,
                                                    self._write_page_access_count)
        
        read_popularity_change_stat = self._get_popularity_change_percentile(read_popularity_map,
                                                                                prev_read_map)
        write_popularity_change_stat = self._get_popularity_change_percentile(write_popularity_map,
                                                                                prev_write_map)
        
        out_str_array = []
        out_str_array += read_popularity_change_stat.get_row()
        out_str_array += write_popularity_change_stat.get_row()

        return out_str_array


    def stat_str_array(self, **kwargs):
        """ This function returns the statistics as an array 
            of strings that can be printed or written to file. """
        
        stat_str_array = []
        stat_str_array += self._snap_req_stats()
        stat_str_array += self._snap_io_stats()
        stat_str_array += self._snap_page_stats()
        stat_str_array += self._snap_seq_stats()
        stat_str_array += self._snap_range_stats()
        stat_str_array += self._snap_alignment_stats()
        stat_str_array += self._snap_working_set_size()
        stat_str_array += self._snap_popularity_stat()
        stat_str_array += self._read_size_stats.get_row()
        stat_str_array += self._write_size_stats.get_row()
        stat_str_array += self._jump_distance_stats.get_row()
        stat_str_array += self._scan_stats.get_row()

        if 'prev_read_map' in kwargs and 'prev_write_map' in kwargs:
            stat_str_array += self._snap_popularity_change_stats(kwargs['prev_read_map'], 
                                                kwargs['prev_write_map'])
        else:
            # TODO: this part not tested yet 
            stat_str_array += PercentileStats().get_row()
            stat_str_array += PercentileStats().get_row()

        assert len(stat_str_array)+2 == len(self.header), \
                    " The length of stat array {} not equal to header array {}".format(len(stat_str_array), len(self.header))
        
        return stat_str_array


    def snapshot(self, 
            window_index, 
            ts, 
            out_handle=None, 
            force_print=False, 
            **kwargs):
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
                force printing even when writing to file (Default: False) """

        if window_index == 0 and out_handle is not None:
            self._write_header(out_handle)

        stat_str_array = self.stat_str_array(**kwargs)
        out_str = ",".join(stat_str_array)
        if out_handle is not None:
            out_handle.write("{},{},{}\n".format(window_index, ts, out_str))
            if force_print:
                print(out_str)
        else:
            print(out_str)

        cur_time = time.time()
        time_elasped_mins = (cur_time-self._start_time)/60
        logger.info("Window: {}, TS: {}, Elasped: {}".format(window_index, 
                                                        ts/1e6,
                                                        time_elasped_mins))


    def add_request(self, block_req):
        """ Update the statistics based on a block request 
            provided by the user. 

            Parameters
            ----------
            block_req : dict 
                dict containing block request features """

        self._track_op_type(block_req)
        self._track_seq_access(block_req)
        self._track_req_alignment(block_req)
        self._track_popularity(block_req)
        self._prev_req = block_req


    def __sub__(self, other):
        """ Override subtraction for this class. 

            Parameters
            ----------
            other : BlockWorkloadStats 
                another BlockWorkloadStats object """

        res = BlockWorkloadStats()
        res._read_block_req_count = self._read_block_req_count - other._read_block_req_count
        res._write_block_req_count = self._write_block_req_count - other._write_block_req_count
        res._read_page_access_count = self._read_page_access_count - other._read_page_access_count
        res._write_page_access_count = self._write_page_access_count - other._write_page_access_count
        res._read_io_request_size_sum = self._read_io_request_size_sum - other._read_io_request_size_sum
        res._write_io_request_size_sum = self._write_io_request_size_sum - other._write_io_request_size_sum
        res._read_seq_count = self._read_seq_count - other._read_seq_count
        res._write_seq_count = self._write_seq_count - other._write_seq_count
        res._read_misalignment_sum = self._read_misalignment_sum - other._read_misalignment_sum
        res._write_misalignment_sum = self._write_misalignment_sum - other._write_misalignment_sum
        res._min_read_page = min(self._min_read_page, other._min_read_page)
        res._min_write_page = min(self._min_write_page, other._min_write_page)
        res._max_read_page = max(self._max_read_page, other._max_read_page)
        res._max_write_page = max(self._max_write_page, other._max_write_page)
        res._read_page_access_counter = self._read_page_access_counter - other._read_page_access_counter
        res._write_page_access_counter = self._write_page_access_counter - other._write_page_access_counter
        res._read_size_stats = self._read_size_stats - other._read_size_stats
        res._write_size_stats = self._write_size_stats - other._write_size_stats
        res._jump_distance_stats = self._jump_distance_stats - other._jump_distance_stats
        res._scan_stats = self._scan_stats - other._scan_stats

        res._prev_req = self._prev_req
        res._prev_window_read_page_access_counter = self._prev_window_read_page_access_counter
        res._prev_window_read_popularity_map = self._prev_window_read_popularity_map
        res._prev_window_write_page_access_counter = self._prev_window_write_page_access_counter
        res._prev_window_write_popularity_map = self._prev_window_write_popularity_map
        res._scan_length = self._scan_length
        res.start_time = self._start_time

        return res 


    def _write_header(self, handle):
        """ This function writes the header to a file represented by a handle. """
        handle.write("{}\n".format(",".join(self.header)))


    def _percentile_header(self, prefix):
        """ This function returns a list of column names for percentile features. """
        out_str = []
        for p in self._percentiles_tracked:
            out_str.append("{}_{}".format(p, prefix))
        return out_str


    def _header(self):
        """ This function returns the header of the data generated by this class. """
        return ["index", "ts"] + self._feature()


    def _feature(self):
        """ This function returns the features of the data generated by this class. """
        feature_array = [
            "block_req_count",
            "read_block_req_count",
            "write_block_req_count",
            "write_block_req_split",
            "io_request_size_sum",
            "read_io_request_size_sum",
            "write_io_request_size_sum",
            "write_io_request_size_split",
            "page_access_count",
            "read_page_access_count",
            "write_page_access_count",
            "write_page_access_split",
            "seq_count",
            "read_seq_count",
            "write_seq_count",
            "write_seq_split",
            "range",
            "read_range",
            "write_range",
            "misalignment_sum",
            "read_misalignment_sum",
            "write_misalignment_sum",
            "page_working_set_size",
            "read_page_working_set_size",
            "write_page_working_set_size",
            "write_page_working_set_size_split"
        ]
        feature_array += self._percentile_header("read_page_popularity")
        feature_array += self._percentile_header("write_page_popularity")
        feature_array += self._percentile_header("read_block_request_size")
        feature_array += self._percentile_header("write_block_request_size")
        feature_array += self._percentile_header("jump_distance")
        feature_array += self._percentile_header("scan_length")
        feature_array += self._percentile_header("delta_read_page_popularity")
        feature_array += self._percentile_header("delta_write_page_popularity")

        return feature_array
