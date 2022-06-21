import numpy as np 
import pathlib 

from pyCydonia.profiler.BlockWorkloadStats import BlockWorkloadStats
from pyCydonia.profiler.RDTracker import RDTracker
from pyCydonia.dataProcessor.cpProcessor import csv_to_page_trace, page_file_to_rd


class BlockTraceProfiler:

    def __init__(self, reader, stat_types=['block'], **kwargs):
        self._page_size = 4096
        self._reader = reader 
        self._workload_name = self._reader.trace_file_path.stem 

        self._stat = {}
        if 'block' in stat_types:
            self._stat['block'] = BlockWorkloadStats()
            if 'workload_snapshot_dir' in kwargs:
                out_dir = pathlib.Path(kwargs['workload_snapshot_dir'])
                out_path = out_dir.joinpath('block_snap_{}.csv'.format(self._workload_name))
                self._workload_stat_snapshot_file_handle = out_path.open('w+')
            else:
                self._workload_stat_snapshot_file_handle = None 

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
        if 'window_size' in kwargs:
            self._window_size = kwargs['window_size']
            if 'rd' in self._stat:
                self._prev_rd_tracker = RDTracker(rd_file=self._rd_file)

        self._cur_req = {}
        self._cur_page_index = -1 
        self._time_elasped = 0 
        

    def _generate_rd_trace(self, page_dir, rd_dir):
        print('Generating RD Trace ..')
        block_trace_path = self._reader.trace_file_path
        file_name = block_trace_path.name 
        page_trace_path = pathlib.Path(page_dir).joinpath(file_name)
        if not page_trace_path.exists():
            csv_to_page_trace(block_trace_path, page_trace_path, self._page_size)
        rd_trace_path = pathlib.Path(rd_dir).joinpath(file_name)
        if not rd_trace_path.exists():
            page_file_to_rd(page_trace_path, rd_trace_path)
        print("RD trace generated: {}!".format(rd_trace_path))
        return rd_trace_path


    def _load_next_block_req(self):
        self._cur_req = self._reader.get_next_block_req(page_size=self._page_size) 
        if self._cur_req:
            self._time_elasped = self._cur_req["ts"]
            self._cur_req["key"] = self._cur_req["start_page"]

            if "block" in self._stat:
                self._stat["block"].add_request(self._cur_req)
            if "rd" in self._stat:
                self._cur_req["rd"] = self._rd_tracker.get_next_rd()


    def _load_next_cache_req(self):
        if not self._cur_req:
            self._load_next_block_req()
        else:
            prev_key = self._cur_req["key"]
            if prev_key == self._cur_req["end_page"]:
                self._load_next_block_req()
            else:
                self._cur_req["key"] += 1
                if "rd" in self._stat:
                    self._cur_req["rd"] = self._rd_tracker.get_next_rd()
                
    
    def cur_window_index(self):
        window_index = 0 
        if (self._cur_req["ts"] > 0): 
            window_index = (self._cur_req["ts"]-1)//self._window_size
        return int(window_index)


    def run(self):
        prev_window_index = -1 
        self._load_next_cache_req()
        while (self._cur_req):

            if "rd" in self._stat:
                assert self._cur_req['rd'] != np.inf, \
                        "The number of page requests and lines in RD trace not equal"

            if self._window_size > 0:

                # window size is defined 
                window_index = self.cur_window_index()
                assert window_index >= prev_window_index, \
                            "Time should always be increasing"

                # when the window changes snapshot stats  
                if window_index != prev_window_index:
                    if prev_window_index >= 0:
                        if "rd" in self._stat:
                            # tracking the RD characteristics 
                            window_rd_tracker = self._rd_tracker - self._prev_rd_tracker
                            window_rd_tracker.snapshot(window_index, 
                                                        self._cur_req["ts"], 
                                                        out_handle=self._rd_stat_snapshot_file_handle,
                                                        force_print=False)
                            self._prev_rd_tracker.copy(self._rd_tracker)

                        if "block" in self._stat: 
                            # tracking the block workload statistics 
                            self._stat["block"].snapshot(window_index,
                                                        self._cur_req["ts"], 
                                                        out_handle=self._workload_stat_snapshot_file_handle,
                                                        force_print=False)

                    prev_window_index = window_index

            self._load_next_cache_req()

        if "rd" in self._stat: 
            assert self._rd_tracker.get_next_rd() == np.inf, \
                    "The number of request in RD trace doesn't align with block trace"
            self._rd_stat_snapshot_file_handle.close()
        
        if "block" in self._stat:
            # self._stat["block"].snapshot(window_index,
            #                             self._cur_req["ts"], 
            #                             out_handle=self._workload_stat_snapshot_file_handle,
            #                             force_print=False)
            if self._workload_stat_snapshot_file_handle is not None:
                self._workload_stat_snapshot_file_handle.close()