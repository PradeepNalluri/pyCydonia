import math 
import time 
import logging 
from collections import namedtuple

import numpy as np 
import matplotlib.pyplot as plt
logging.basicConfig(format='%(asctime)s,%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

from pyCydonia.cacheServer.cacheServer import CacheServer

class LoadDataException(Exception):
    pass


class NPHist:
    def __init__(self, PAGE_SIZE=4096):
        self.page_size = PAGE_SIZE
        # hit stats 
        self.hit_count = np.empty(0)
        self.cold_miss_count = [0, 0]
        # cache info 
        self.max_cache_size = 0
        self.max_cache_size_mb = 0
        # IO stat
        self.read_count = 0 
        self.write_count = 0 
        self.io_count = 0 
        

    def load(self, rd_hist_file):
        """ self.hit_count contains the number of read and write 
            hits at cache sizes equal to the index. So, self.rd_hist[2]
            contains the number of read and write hits for cache size 2. 
            self.rd_hist[0] = [0,0] 
        """

        file_data = np.genfromtxt(rd_hist_file, delimiter=',', dtype=int)
        self.hit_count = np.zeros((len(file_data), 2), dtype=int)

        # not storing cold misses from the file data at file_data[0] in hit count 
        self.hit_count[1:] = file_data[1:].cumsum(axis=0) 
        self.cold_miss_count = file_data[0]

        self.max_cache_size = len(file_data)-1 
        self.max_cache_size_mb = ((len(file_data)-1)*self.page_size)/(1024*1024) # convert to MB 
        
        self.read_count = self.hit_count[-1][0] + self.cold_miss_count[0]
        self.write_count = self.hit_count[-1][1] + self.cold_miss_count[1]
        self.io_count = self.read_count + self.write_count


    def read_cold_miss_rate(self):
        if self.hit_count.size == 0:
            raise LoadDataException("No Rd Histogram loaded!")
        return self.cold_miss_count[0]/self.read_count


    def write_percent(self):
        if self.hit_count.size == 0:
            raise LoadDataException("No Rd Histogram loaded!")
        return 100*self.write_count/self.io_count

    
    def plot_read_mrc(self, 
        output_path, 
        num_x_labels=10, 
        cache_size_multiple=25, 
        max_mrc_size_mb=2048):

        if self.hit_count.size == 0:
            raise LoadDataException("No Rd Histogram loaded!")

        r_mrc = (self.read_count - self.hit_count[:, 0])/self.read_count
        MRC_LENGTH = len(r_mrc)
        if max_mrc_size_mb > 0:
            max_cache_size = min(math.ceil((max_mrc_size_mb*1024*1024)/self.page_size), len(r_mrc))
            MRC_LENGTH = min(max_cache_size, len(r_mrc))

        fig, ax = plt.subplots(figsize=(14,7))
        ax.plot(r_mrc[:MRC_LENGTH])

        # setup x-axis labels
        xtick_values = np.arange(0, MRC_LENGTH+1, (cache_size_multiple*1024*1024)/self.page_size)
        xtick_values_scaled = xtick_values[0::int(len(xtick_values)/num_x_labels)]
        xlabel_ticks = ['{}'.format(int(_ * self.page_size/(1024*1024))) for _ in xtick_values_scaled]
        ax.tick_params(axis='x', rotation=90)
        ax.set_xticks(xtick_values_scaled)
        ax.set_xticklabels(xlabel_ticks)

        label_format = '{}'
        xlabel_ticks = [label_format.format(int(_)) for _ in np.linspace(256, MRC_LENGTH, num=10)//256]
        
        ax.set_xlabel("Cache Size (MB)")
        ax.set_ylabel("Miss Rate")
        ax.set_title("Workload: {}, Read Cold Miss Rate: {:.2f} \n Ops: {:.1f}M / {}GB Write: {:.1f}%".format(
            output_path.stem, 
            self.read_cold_miss_rate(),
            self.io_count/1000000,
            math.ceil(self.io_count*self.page_size/(1024*1024*1024)),
            self.write_percent()))

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()



    def get_exclusive_hits(self, t1_size, t2_size):
        t1_hits = np.sum(self.hit_count[:t1_size+1][0], axis=1)
        t2_hits = np.sum(self.hit_count[t1_size+1:t1_size+t2_size+1])
        return t1_hits, t2_hits 










    def get_lat_ratio_curve(self, cache_server, budget_percentage_array=range(1,101),
        min_allocation_size=1):
        """ Generate an OPT latency ratio curve for this RD histogram and cache server. 
        """
        max_budget = cache_server.config[0]["price"]*self.max_cache_size
        min_lat = self.get_min_lat(cache_server)
        output_tuple = namedtuple('output_tuple', self.output_headers)

        st_data = []
        lat_ratio_curve = []
        for budget_percentage in budget_percentage_array:
            start = time.time()
            budget_ratio = float(budget_percentage/100)
            cur_budget = budget_ratio * max_budget
            max_t1_size = math.floor(cur_budget/(cache_server.config[0]["price"]*min_allocation_size))
            max_t2_size = math.floor(cur_budget/(cache_server.config[1]["price"]*min_allocation_size))

            # eval single tier configurations first and record 
            d1_st_cache_hit_stats, d1_lat_penalty = self.eval_cache([max_t1_size*min_allocation_size, 0], cache_server)
            d2_st_cache_hit_stats, d2_lat_penalty = self.eval_cache([0, max_t2_size*min_allocation_size], cache_server)
            d1_lat_ratio = min_lat/d1_lat_penalty
            d2_lat_ratio = min_lat/d2_lat_penalty
            max_lat_ratio = max(d1_lat_ratio, d2_lat_ratio)
            max_lat_ratio_mt_config = [max_t1_size*min_allocation_size, 0] if d1_lat_ratio >= d2_lat_ratio \
                else [0, max_t2_size*min_allocation_size]
            max_lat_ratio_cache_hit_stats = d1_st_cache_hit_stats if d1_lat_ratio >= d2_lat_ratio \
                else d2_st_cache_hit_stats
            max_lat_ratio_budget_split = 1.0 if d1_lat_ratio >= d2_lat_ratio else 0.0
            d1_output_tuple = output_tuple(budget_ratio, cur_budget, d1_lat_ratio, 1.0, 
                max_t1_size, 0)
            d2_output_tuple = output_tuple(budget_ratio, cur_budget, d2_lat_ratio, 0.0, 
                0, max_t2_size)
            st_data.append([d1_output_tuple, d2_output_tuple])

            for t1_size in range(max_t1_size+1):
                cur_t1_size = t1_size * min_allocation_size
                cur_t1_budget = cur_t1_size*cache_server.config[0]["price"]
                cur_t2_budget = cur_budget - cur_t1_budget
                cur_t2_size = math.floor(cur_t2_budget/cache_server.config[1]["price"])

                if cur_t2_budget < cache_server.config[1]["price"]:
                    cur_t2_budget = 0
                    cur_t2_size = 0

                assert cur_t1_budget>=0 and cur_t2_budget>=0, \
                    "Tier 1 Budget: {}, Tier 2 Budget: {}, Tier with negative budget!".format(
                        cur_t1_budget, cur_t2_budget)

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
            "Negative cache size found. L1 {}, L2 {}, max cache size: {}, tiers: {}".format(
                adjusted_l1_size, adjusted_l2_size, self.max_cache_size, tiers)

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