import numpy as np 
import pathlib 
import copy 

from collections import Counter


class RDTracker:
    def __init__(self, **kwargs):
        if 'rd_file' in kwargs:
            self._rd_file = pathlib.Path(kwargs['rd_file'])
            self._handle = open(self._rd_file, 'r')

        self._req_count = 0 
        self.max_rd = -1 
        self._rd_counter = Counter()
        self.composed = False
        
        self._min_hit_rate = 0.1
        self._max_hit_rate = 1.0
        self._hit_rate_step = 0.1


    def add_rd(self, rd):
        self._req_count += 1
        self._rd_counter[rd] += 1
        if rd > self.max_rd:
            self.max_rd = rd 


    def get_next_rd(self):
        if self.composed:
            raise ValueError("This RDTracker object was composed from two other RDTracker objects. Cannot load by file.")
        else:
            line = self._handle.readline()
            if line:
                rd = int(line.split(",")[0])
                self.add_rd(rd)
            else:
                rd = np.inf
        
        return rd 


    def snapshot(self, window_index, ts, out_handle=None, force_print=False):
        min_cache_size = self.get_min_cache_size_array(np.arange(0.01,1.0,0.01))
        out_str = "{},{},{},{}\n".format(window_index, ts, self._req_count, ",".join([str(_) for _ in min_cache_size]))
        if out_handle is not None:
            out_handle.write(out_str)
            if force_print:
                print(out_str)
        else:
            print(out_str)


    def get_min_cache_size_array(self, hit_rate_array):
        """ Get an array of minimum size of cache needed to get a hit rate more
            than the each hit rate in the hit_rate_array which is assumed 
            to be sorted. 
        """

        hit_rate_array_len = len(hit_rate_array)
        min_cache_size_array = np.full(hit_rate_array_len + 1, -1, dtype=int)
        cur_hit_rate_index = 0 
        hit_count = 0
        break_flag = False 
        for cache_size in range(1, self.max_rd+1):
            hit_count += self._rd_counter[cache_size-1]
            
            hit_rate = 0.0 
            if self._req_count > 0:
                hit_rate = hit_count/self._req_count

                while (hit_rate > hit_rate_array[cur_hit_rate_index]):
                    min_cache_size_array[cur_hit_rate_index] = cache_size 
                    cur_hit_rate_index += 1

                    if cur_hit_rate_index == hit_rate_array_len-1:
                        break_flag = True
                        break
            
            if break_flag:
                min_cache_size_array[hit_rate_array_len] = self.max_rd+1
                break 
        
        return min_cache_size_array


    def __sub__(self, other):
        new = RDTracker()
        new.composed = True 
        new._req_count = self._req_count - other._req_count
        new._rd_counter = self._rd_counter - other._rd_counter 

        # get the max RD of the new RDTracker 
        higher_max_rd = max(self.max_rd, other.max_rd)
        max_rd = higher_max_rd 
        for rd in range(higher_max_rd+1):
            if new._rd_counter[rd] > 0:
                max_rd = rd 
        new.max_rd = max_rd 
        return new 


    def copy(self, other):
        self.composed = True 
        self._req_count = other._req_count
        self.max_rd = other.max_rd 
        self._rd_counter = copy.deepcopy(other._rd_counter)
