import numpy as np 
import pathlib 
import copy 
import time 
import logging
from collections import defaultdict
logging.basicConfig(format='%(levelname)s:%(message)s')
logger = logging.getLogger('profiler_logger')
logger.setLevel(logging.DEBUG)

from pyCydonia.profiler.BlockWorkloadStats import BlockWorkloadStats
from pyCydonia.profiler.RDTracker import RDTracker
from pyCydonia.dataProcessor.cpProcessor import csv_to_page_trace, page_file_to_rd

""" BlockTraceProfiler
    ------------------
    This class profiles a block trace and generates features. 

    Parameters
    ----------
    reader : Reader
        a reader that returns cache requests in order 
    stat_types : list
        list of strings indicating what types of features to generate 
"""
class BlockTraceProfiler:

    def __init__(self, reader, stat_types=['block'], **kwargs):
        self._page_size = 4096
        self._reader = reader 
        self._workload_name = self._reader.trace_file_path.stem 

        self._stat = {} # current stat
        self._stat_local = {} # stat of the current window 
        self._prev_stat_local = {} # stat of the previous window 

        # track block workload features 
        if 'block' in stat_types:
            self._stat['block'] = BlockWorkloadStats()
            self._stat_local['block'] = BlockWorkloadStats()
            self._prev_stat_local['block'] = None 

            # whether to snapshot stats at a fixed interval of 30 minutes 
            if 'snapshot_dir' in kwargs:
                out_dir = pathlib.Path(kwargs['snapshot_dir'])
                out_path = out_dir.joinpath('block_snap_{}.csv'.format(self._workload_name))
                self._workload_stat_snapshot_file_handle = out_path.open('w+')
            else:
                self._workload_stat_snapshot_file_handle = None 

        # track reuse distance features 
        if 'rd' in stat_types:
            self._rd_tracker = None
            if 'rd_file' in kwargs:
                self._rd_file = kwargs['rd_file']
            else:
                assert 'page_dir' in kwargs and 'rd_dir' in kwargs, \
                            'page_dir and rd_dir have to be set in kwargs if rd_file not set'
                self._rd_file = self._generate_rd_trace(kwargs['page_dir'], 
                                                        kwargs['rd_dir'])
        
            if 'rd_snapshot_dir' in kwargs:
                self._rd_out_dir = pathlib.Path(kwargs['rd_snapshot_dir'])
                rd_out_path = self._rd_out_dir.joinpath('{}.csv'.format(self._workload_name))
                self._rd_stat_snapshot_file_handle = rd_out_path.open('w+')
        
            self._rd_tracker = RDTracker(rd_file=self._rd_file)
        
        # collect stats at different window sizes 
        self._window_size = 0
        self._window_start_time = 0 
        self._window_count = 0 
        self._max_window_index = -1
        if 'window_size' in kwargs:
            self._window_size = kwargs['window_size']
            if 'rd' in self._stat:
                self._prev_rd_tracker = RDTracker(rd_file=self._rd_file)

        self._cur_req = {}
        self._cur_page_index = -1 
        self._time_elasped = 0 
        self._prev_req = {}
        

    def _generate_rd_trace(self, page_dir, rd_dir):
        """ This function generates a page request trace and 
            reuse distance trace from a block trace. 

            Parameters
            ----------
            page_dir : str 
                path to directory with page request traces
            rd_dir : str 
                path to directory with reuse distance traces 

            Return 
            ------
            rd_trace_path : str 
                path to reuse distance trace 
        """
        block_trace_path = self._reader.trace_file_path
        file_name = block_trace_path.name 
        page_trace_path = pathlib.Path(page_dir).joinpath(file_name)
        if not page_trace_path.exists():
            logger.info('Generating page trace {}'.format(page_trace_path))
            csv_to_page_trace(block_trace_path, page_trace_path, self._page_size)
        rd_trace_path = pathlib.Path(rd_dir).joinpath(file_name)
        if not rd_trace_path.exists():
            logger.info('Generating RD trace {}'.format(rd_trace_path))
            page_file_to_rd(page_trace_path, rd_trace_path)
        logger.info('Page and RD trace generated from block trace!')
        return rd_trace_path


    def _load_next_block_req(self):
        """ This function returns the next block request from the reader. """
        self._cur_req = self._reader.get_next_block_req(page_size=self._page_size) 
        if self._cur_req:
            self._time_elasped = self._cur_req["ts"]
            self._cur_req["key"] = self._cur_req["start_page"]
            if "rd" in self._stat:
                self._cur_req["rd"] = self._rd_tracker.get_next_rd()


    def _load_next_cache_req(self):
        """ This function returns the next cache request. """
        self._prev_req = self._cur_req
        if not self._cur_req:
            self._load_next_block_req()
        else:
            if self._cur_req["key"] == self._cur_req["end_page"]:
                self._load_next_block_req()
            else:
                self._cur_req["key"] += 1
                if "rd" in self._stat:
                    self._cur_req["rd"] = self._rd_tracker.get_next_rd()
                
    
    def cur_window_index(self):
        """ This function returns the index of the window based on timestamp. """
        window_index = 0 
        if (self._cur_req["ts"] > 0): 
            window_index = (self._cur_req["ts"]-1)//self._window_size
        return int(window_index)


    def _snap_stats(self, window_index):
        snapshot_kwargs = {}
        if self._prev_stat_local['block'] is not None:
            snapshot_kwargs['prev_read_map'] = self._prev_stat_local['block'].read_page_popularity_map()
            snapshot_kwargs['prev_write_map'] = self._prev_stat_local['block'].write_page_popularity_map()
        else:
            snapshot_kwargs['prev_read_map'] = defaultdict(float)
            snapshot_kwargs['prev_write_map'] = defaultdict(float)

        self._stat_local['block'].snapshot(window_index,
                                        self._prev_req["ts"], 
                                        out_handle=self._workload_stat_snapshot_file_handle,
                                        force_print=False,
                                        **snapshot_kwargs)
                    
        self._prev_stat_local['block'] = copy.deepcopy(self._stat_local['block'])
        self._stat_local['block'] = BlockWorkloadStats()


    def _snap_rd(self):
        pass 


    def generate_features(self, out_path=None):
        """ This function computes features from the provided trace. """
        start_time = time.time()
        # window index begins from 0
        prev_window_index = -1 
        self._load_next_cache_req()
        while (self._cur_req):
            # check if collecting stats per window is enabled 
            if self._window_size > 0:
                window_index = self.cur_window_index()
                assert window_index >= prev_window_index, \
                            "Time should always be increasing"

                # when the window changes snapshot stats  
                while window_index != prev_window_index:
                    # if the previous window is -1 we can ignore 
                    if prev_window_index >= 0:
                        if 'rd' in self._stat:
                            window_rd_tracker = self._rd_tracker - self._prev_rd_tracker
                            window_rd_tracker.snapshot(prev_window_index, 
                                                        self._prev_req["ts"], 
                                                        out_handle=self._rd_stat_snapshot_file_handle,
                                                        force_print=False)
                            self._prev_rd_tracker.copy(self._rd_tracker)

                        if 'block' in self._stat: 
                            self._max_window_index = prev_window_index
                            self._snap_stats(prev_window_index)
                    prev_window_index += 1
                
            if 'block' in self._stat:
                self._stat['block'].add_request(self._cur_req)
                self._stat_local['block'].add_request(self._cur_req)
            self._load_next_cache_req()

        if 'rd' in self._stat: 
            assert self._rd_tracker.get_next_rd() == np.inf, \
                    "The number of request in RD trace doesn't align with block trace"
            self._rd_stat_snapshot_file_handle.close()
        
        if 'block' in self._stat:
            if self._stat_local['block'].block_req_count() > 0:
                self._snap_stats(self._max_window_index+1)
            if self._workload_stat_snapshot_file_handle is not None:
                self._workload_stat_snapshot_file_handle.close()

            # print final stats 
            stat_str_array = self._stat['block'].stat_str_array()
            feature_array = self._stat['block'].features
            for feature_index, feature_header in enumerate(feature_array):
                logger.info("{}: {}".format(feature_header, stat_str_array[feature_index]))

            # if we output to a file 
            if out_path is not None:
                if not pathlib.Path(out_path).exists():
                    with open(out_path, 'w+') as f:
                        f.write("")
                with open(out_path, "r+") as f:
                    data = f.read()
                    if len(data) == 0:
                        f.write("workload,{}\n".format(",".join(feature_array)))
                    f.write("{},{}\n".format(self._workload_name, ",".join(stat_str_array)))
                f.close()
        end_time = time.time()
        time_elasped_mins = (end_time-start_time)/60
        logger.info("Runtime: {}".format(time_elasped_mins))