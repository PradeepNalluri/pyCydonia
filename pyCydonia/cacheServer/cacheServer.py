import math, os 
import numpy as np 
import pandas as pd 
from collections import namedtuple

class CacheServer():

    def __init__(self, config):
        self.config = config
        self.num_tiers = len(config)
        self.exclusive_wb_latency = self.get_exclusive_wb_mt_cache_latency()
        self.st_latency = [self.get_wb_st_cache_latency(i) for i in range(len(config)-1)]


    def get_wb_st_cache_latency(self, device_index) -> np.ndarray:
        return self.get_exclusive_wb_mt_cache_latency([self.config[device_index], self.config[-1]])


    def get_exclusive_wb_mt_cache_latency(self, cache_config=None) -> np.ndarray:
        """ Returns the latency of a exclusive, write-back multi-tier cache. 
        
        Computes latency of each tier of an exclusive multi-tier
        cache using write-back policy using devices in each tier. 

        Args:
            self: A CacheServer instance. 

        Returns:
            A [2,3] numpy array where the first index represents read (0) and 
            write(1). The second index represents tier 1 hit latency (0), tier 
            2 hit latency (1), miss latency (2) respectively. For example: 
            
            [[1,4,10],[2,5,20]] where
            1 and 2 are the latency of read and write hit of tier 1.
            4 and 5 are the latency of read and write hit of tier 2.
            10 and 20 are the latency of read and write miss. 
        """

        if cache_config is None:
            cache_config = self.config

        lat_array = np.zeros([2, len(cache_config)], dtype=float)
        cur_read_latency = 0.0
        cur_write_latency = 0.0 
        for tier_index, tier_device in enumerate(cache_config):
            if tier_index == 0:
                """ The top tier of the cache will have latency equal to the device 
                    read and write latency because there is no eviction.
                """
                lat_array[0][tier_index] = cache_config[tier_index]["read_lat"]
                lat_array[1][tier_index] = cache_config[tier_index]["write_lat"]
            elif tier_index < len(cache_config)-1:
                """ On a read hit, there is a read and write in every tier including the 
                    one where the read hit happened (read for the page requested and write 
                    for the eviction from the tier above). 
                    
                    On a write hit, there is a read and write in every tier above it and 
                    only a write where the data to be overwritten was because there is no 
                    eviction from that tier as the page can be simply overwritten 
                    to fit the page that is evicted from the tier above. 
                """
                lat_array[0][tier_index] = cache_config[tier_index-1]["read_lat"] + tier_device["read_lat"] \
                    + tier_device["write_lat"]
                lat_array[1][tier_index] = cache_config[tier_index-1]["read_lat"] +tier_device["write_lat"]
            else:
                """ On a read miss, there is a read and write in every tier along with a read from storage.
                    This is the sum of read latency of the last tier of cache and latency of the storage device. 

                    A write miss has the same latency as a write hit in the last cache tier because it causes a
                    read and write in every cache tier except the last as the last tier does not need to flush 
                    data to the next tier. We assume dirty pages are synced asynchronously and do not account for it 
                    in the latency experienced by a tenant. 
                """
                lat_array[0][-1] = lat_array[0][-2] + cache_config[-1]["read_lat"]
                lat_array[1][-1] = lat_array[1][-2] 

        return lat_array