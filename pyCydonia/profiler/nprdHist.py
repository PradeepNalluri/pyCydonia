import math 
import argparse 
import time 
import pathlib 
import logging
import constant
from collections import namedtuple
import numpy as np 
logging.basicConfig(format='%(asctime)s,%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

from pyCydonia.cacheServer.cacheServer import CacheServer

class NPHist:
    def __init__(self):
        self.data = None
        self.cold_miss = None 
        self.max_cache_size = 0
        self.output_headers = ['budget_ratio', 'cur_budget',
            'lat_ratio', 'budget_split', 't1_size', 't2_size']


    def load_rd_hist_file(self, rd_hist_file):
        file_data = np.genfromtxt(rd_hist_file, delimiter=',', dtype=int)

        """ self.data contains the number of read and write hits for cache 
            sizes represented by the index. self.data[2] gives the number 
            of read and write hits for an LRU cache of size 2. 
        """
        self.data = np.zeros((len(file_data), 2), dtype=int)
        self.max_cache_size = len(file_data)-1
        self.data[1:] = file_data[1:].cumsum(axis=0)
        self.cold_miss = file_data[0]


    def get_lat_ratio_curve(self, cache_server, budget_percentage_array=range(1,101)):
        """ Generate an OPT latency ratio curve for this RD histogram and cache server. 
        """
        max_budget = cache_server.config[0]["price"]*len(self.data)
        min_lat = self.get_min_lat(cache_server)
        output_tuple = namedtuple('output_tuple', self.output_headers)

        st_data = []
        lat_ratio_curve = []
        for budget_percentage in budget_percentage_array:
            start = time.time()
            budget_ratio = float(budget_percentage/100)
            cur_budget = budget_ratio * max_budget
            max_t1_size = math.floor(cur_budget/cache_server.config[0]["price"])
            max_t2_size = math.floor(cur_budget/cache_server.config[1]["price"])

            # eval single tier configurations first and record 
            d1_st_cache_hit_stats, d1_lat_penalty = self.eval_cache([max_t1_size, 0], cache_server)
            d2_st_cache_hit_stats, d2_lat_penalty = self.eval_cache([0, max_t2_size], cache_server)
            d1_lat_ratio = min_lat/d1_lat_penalty
            d2_lat_ratio = min_lat/d2_lat_penalty
            max_lat_ratio = max(d1_lat_ratio, d2_lat_ratio)
            max_lat_ratio_mt_config = [max_t1_size, 0] if d1_lat_ratio >= d2_lat_ratio \
                else [0, max_t2_size]
            max_lat_ratio_cache_hit_stats = d1_st_cache_hit_stats if d1_lat_ratio >= d2_lat_ratio \
                else d2_st_cache_hit_stats
            max_lat_ratio_budget_split = 1.0 if d1_lat_ratio >= d2_lat_ratio else 0.0
            d1_output_tuple = output_tuple(budget_ratio, cur_budget, d1_lat_ratio, 1.0, 
                max_t1_size, 0)
            d2_output_tuple = output_tuple(budget_ratio, cur_budget, d2_lat_ratio, 0.0, 
                0, max_t2_size)
            st_data.append([d1_output_tuple, d2_output_tuple])

            for cur_t1_size in range(max_t1_size+1):
                cur_t1_budget = cur_t1_size * cache_server.config[0]["price"]
                cur_t2_budget = cur_budget - cur_t1_budget
                cur_t2_size = math.floor(cur_t2_budget/cache_server.config[1]["price"])

                cache_hit_stats, lat_penalty = self.eval_cache([cur_t1_size, cur_t2_size], cache_server)
                lat_ratio = min_lat/lat_penalty

                if lat_ratio > max_lat_ratio:
                    max_lat_ratio = lat_ratio 
                    max_lat_ratio_mt_config = [cur_t1_size, cur_t2_size]
                    max_lat_ratio_cache_hit_stats = cache_hit_stats
                    max_lat_ratio_budget_split = cur_t1_budget/(cur_t1_budget+cur_t2_budget)

            cur_output_tuple = output_tuple(budget_ratio, cur_budget, max_lat_ratio, max_lat_ratio_budget_split, \
                max_lat_ratio_mt_config[0], max_lat_ratio_mt_config[1])
            lat_ratio_curve.append(cur_output_tuple)
            end = time.time()
            logging.info("Budget Ratio: {}, Lat Ratio: {}, Time: {}".format(budget_ratio, lat_ratio, end-start))

        return lat_ratio_curve, st_data


    def get_min_lat(self, cache_server):
        """ Get the minimum possible latency in the cache server. 
        """
        rw_count = self.data[-1]
        return np.sum(np.multiply(cache_server.st_latency[0],
            np.array([[rw_count[0], self.cold_miss[0]], [rw_count[1], self.cold_miss[1]]])))


    def eval_cache(self, tiers, cache_server):
        """ Evaluate a cache in a cache server with the given tiers.
        """
        adjusted_l1_size = min(tiers[0], self.max_cache_size)
        adjusted_l2_size = min(tiers[1], self.max_cache_size-adjusted_l1_size)

        assert adjusted_l1_size>=0 and adjusted_l2_size>=0, \
            "Negative cache size found. L1 {}, L2 {}".format(adjusted_l1_size, adjusted_l2_size)

        if tiers[0] == 0:
            cache_hit_stats = self.get_cache_hit_stats([adjusted_l2_size])
            lat_penalty = np.sum(np.multiply(cache_hit_stats, cache_server.st_latency[1]))
        elif tiers[1] == 0:
            cache_hit_stats = self.get_cache_hit_stats([adjusted_l1_size])
            lat_penalty = np.sum(np.multiply(cache_hit_stats, cache_server.st_latency[0]))
        else:
            cache_hit_stats = self.get_cache_hit_stats([adjusted_l1_size, adjusted_l2_size])
            lat_penalty = np.sum(np.multiply(cache_hit_stats, cache_server.exclusive_wb_latency))

        return cache_hit_stats, lat_penalty 


    def get_cache_hit_stats(self, cache):
        """ Return the read/write cache hit statistics for each tier of a cache. 

            The stat array contains the following information for an n tier cache. 

            Row 0: tier_0 read hits, tier_1 read hits .... tier_n read hits, read_miss
            Row 1: tier_0 write hits, tier_1 write hits .... tier_n write hits, write_miss
        """
        cache_hit_stats = np.zeros([2, len(cache)+1], dtype=int)

        """ Track the previous read, write hits to subtract from the current cumulative
            read, write hits in order to get the number of hits in each cache tier. 
        """
        prev_cum_read_hits = 0 
        prev_cum_write_hits = 0 
        cur_cache_size = 0 
        for tier_index, cache_tier_size in enumerate(cache):
            cur_cache_size += cache_tier_size
            cache_hit_stats[0][tier_index] = self.data[cur_cache_size][0] - prev_cum_read_hits
            cache_hit_stats[1][tier_index] = self.data[cur_cache_size][1] - prev_cum_write_hits
            prev_cum_read_hits = self.data[cur_cache_size][0]
            prev_cum_write_hits = self.data[cur_cache_size][1]
        else:
            cache_hit_stats[0][-1] = self.data[-1][0] - prev_cum_read_hits + self.cold_miss[0]
            cache_hit_stats[1][-1] = self.data[-1][1] - prev_cum_write_hits + self.cold_miss[1]

        return cache_hit_stats