# coding=utf-8

"""
    The class represents the reuse distance histograms. It allows users to do operations such
    as add or subtract histograms and generate features.

    Author: Pranav Bhandari <bhandaripranav94@gmail.com> 2020/11
"""

import math 
import numpy as np 
import pandas as pd 
from collections import Counter, defaultdict

from pyCydonia.cacheServer.cacheServer import CacheServer

class RDHist:

    def __init__(self):
        self.r = Counter()
        self.w = Counter()
        self.max_rd = -1 


    def load_rd_file(self, rd_file_path, rd_index=0, op_index=1, delimiter=","):
        """ Load RD file and generate the relevant read and write counter for RD values. 
        """

        with open(rd_file_path) as f:
            line = f.readline().rstrip()
            while line:
                split_line = line.split(delimiter)
                rd = int(split_line[rd_index])
                op = split_line[op_index]
                self.update(rd, op=op)
                line = f.readline().rstrip()


    def load_rd_hist_file(self, rd_hist_file_path, delimiter=","):
        """ Load a reuse distance histogram from a RD histogram file. 
        """

        with open(rd_hist_file_path) as f:
            cur_rd = -1
            line = f.readline().rstrip()
            while line:
                split_line = line.split(delimiter)
                self.r[cur_rd] = int(split_line[0])
                self.w[cur_rd] = int(split_line[1])
                cur_rd += 1 
                line = f.readline().rstrip()

            if cur_rd-1 > self.max_rd:
                self.max_rd = cur_rd 


    def update_rd_hist_file(self, rd_hist_file_path, delimiter=","):
        """ Update the current RDHist with data from another reuse distance histogram. 
        """

        with open(rd_hist_file_path) as f:
            cur_rd = -1
            line = f.readline().rstrip()
            while line:
                split_line = line.split(delimiter)
                self.r[cur_rd] += int(split_line[0])
                self.w[cur_rd] += int(split_line[1])
                cur_rd += 1 
                line = f.readline().rstrip()

            if cur_rd-1 > self.max_rd:
                self.max_rd = cur_rd 


    def update(self, rd, op="r"):
        """ Update the count in the read (r) or write (w) histogram
        """

        if op == "r":
            self.r.update([rd])
        elif op == "w":
            self.w.update([rd])
        else:
            raise ValueError('The operation is not "r" or "w".')
        if rd > self.max_rd: 
            self.max_rd = rd


    def write_to_file(self, output_file):
        """ Generate the reuse distance histogram file which can be used to load 
            the histogram to obtain this object. 
        """

        with open(output_file, "w+") as f:
            for i in range(-1, self.max_rd+1):
                f.write("{},{}\n".format(self.r[i], self.w[i]))


    def get_opt_mt_exclusive_wb_for_budget_and_devices(self, budget, devices, output_file=None):
        """ Get the optimal exclusive write-back multi-tier cache for the given budget
            and device choices. 

            This function returns 3 elements:
                min_config: array containg the optimal size in MB of tier 1 and tier 2 
                min_config_stat: array containing read and write hits for each tier and misses 
                min_lat: the latency of the OPT config 
        """

        if output_file is not None:
            output_file_handle = open(output_file, "w+")

        max_tier_1_size_mb = math.floor(budget/(devices[0]["price"]*256)) 
        min_lat, min_config, min_config_stat = math.inf, None, None 

        # workload info needed 
        cum_rd_count_array = self.get_cumulative_rd_count_array()
        read_cold_miss_count, write_cold_miss_count = self.get_cold_miss_count()

        # cache server info needed 
        cache_server = CacheServer(devices)
        exclusive_wb_latency_array = cache_server.get_exclusive_wb_mt_cache_latency()

        for cur_tier_1_size_mb in range(max_tier_1_size_mb+1):
            
            remaining_budget = budget - cur_tier_1_size_mb*(devices[0]["price"]*256)
            cur_tier_2_size_mb = min(math.floor(remaining_budget/(devices[1]["price"]*256)), 
                math.floor(self.max_rd/256))

            if self.max_rd < 256 and math.floor(remaining_budget/(devices[1]["price"]*256))>0:
                cur_tier_2_size_mb = 1

            # adjust tier 1 and tier 2 sizes 
            cur_tier_1_size = cur_tier_1_size_mb * 256 if cur_tier_1_size_mb*256 <= self.max_rd+1 else self.max_rd+1
            cur_tier_2_size = cur_tier_2_size_mb * 256 if (cur_tier_1_size_mb+cur_tier_2_size_mb)*256 <= self.max_rd+1 \
                else self.max_rd+1-cur_tier_1_size

            if cur_tier_1_size == 0 and cur_tier_2_size == 0:
                assert 1==0, "The current max rd is {}, budget {}, rem budget {}".format(self.max_rd,
                    budget, remaining_budget)

            if self.max_rd == -1:
                mt_stat_array = np.array([[0,0,read_cold_miss_count],[0,0,write_cold_miss_count]])
            else:
                mt_stat_array = self.get_mt_stat_from_cumulative_rd_count(cum_rd_count_array, 
                    [read_cold_miss_count, write_cold_miss_count], cur_tier_1_size, cur_tier_2_size)

            cur_lat_array = np.multiply(mt_stat_array, exclusive_wb_latency_array)
            cur_mean_lat = np.sum(cur_lat_array)/np.sum(mt_stat_array)

            if output_file is not None:
                output_file_handle.write("{},{},{},{}\n".format(
                    cur_tier_1_size,
                    cur_tier_2_size,
                    (budget-remaining_budget)/budget,
                    cur_mean_lat
                ))

            if cur_mean_lat < min_lat:
                min_lat = cur_mean_lat
                min_config = [cur_tier_1_size_mb, cur_tier_2_size_mb]
                min_config_stat = mt_stat_array
        else:
            cur_tier_1_size_mb = max_tier_1_size_mb
            cur_tier_2_size_mb = 0
            cur_tier_1_size = max_tier_1_size_mb*256 if max_tier_1_size_mb*256 <= self.max_rd+1 else self.max_rd+1
            cur_tier_2_size = cur_tier_2_size_mb*256
            # mt_stat_array = self.get_mt_stat_from_cumulative_rd_count(cum_rd_count_array, 
            #     [read_cold_miss_count, write_cold_miss_count], cur_tier_1_size, cur_tier_2_size)

            if self.max_rd == -1:
                mt_stat_array = np.array([[0,0,read_cold_miss_count],[0,0,write_cold_miss_count]])
            else:
                mt_stat_array = self.get_mt_stat_from_cumulative_rd_count(cum_rd_count_array, 
                    [read_cold_miss_count, write_cold_miss_count], cur_tier_1_size, cur_tier_2_size)

            cur_lat_array = np.multiply(mt_stat_array, exclusive_wb_latency_array)
            cur_mean_lat = np.sum(cur_lat_array)/np.sum(mt_stat_array)

            # print("Evaluting TIER 1 size {} for lat {} and min lat {}".format(cur_tier_1_size, cur_mean_lat, min_lat))

            if output_file is not None:
                output_file_handle.write("{},{},{},{}\n".format(
                    cur_tier_1_size,
                    cur_tier_2_size,
                    (budget-remaining_budget)/budget,
                    cur_mean_lat
                ))

            if cur_mean_lat < min_lat:
                min_lat = cur_mean_lat
                min_config = [cur_tier_1_size_mb, cur_tier_2_size_mb]
                min_config_stat = mt_stat_array
        

        if output_file is not None:
            output_file_handle.close()

        return min_config, min_config_stat, min_lat 


    def get_cumulative_rd_count_array(self):
        """ Get array with cumulative count for the RD histogram. Cumulative counts are useful when 
            using the same file over and over to evaluate different configurations as it saves time
            because you don't have to scan through the reuse distance counts and add them up everytime.
        """

        cum_rd_count_array = np.zeros([self.max_rd+1, 2], dtype=int)
        cum_rd_count_array[0] = np.array([self.r[0], self.w[0]])
        for cur_rd in range(1, self.max_rd+1):
            cum_rd_count_array[cur_rd] = np.array([cum_rd_count_array[cur_rd-1][0]+self.r[cur_rd], 
                cum_rd_count_array[cur_rd-1][1]+self.w[cur_rd]])
        return cum_rd_count_array

    
    def evaluate_mt_exclusive_wb(self, config, devices, cum_rd_count_array=None):
        """ Evaluate a given multi-tier configuration for an exclusive, write-back cache for the given 
            device performance and optional cumulative RD count array. If the cumulative RD count array
            is not provided, it will generate one for this RD histogram at this point of time and use it.
        """

        cache_server = CacheServer(devices)
        exclusive_wb_latency_array = cache_server.get_exclusive_wb_mt_cache_latency()
        read_cold_miss_count, write_cold_miss_count = self.get_cold_miss_count()

        if cum_rd_count_array is None:
            cum_rd_count_array = self.get_cumulative_rd_count_array()

        mt_stat_array = self.get_mt_stat_from_cumulative_rd_count(cum_rd_count_array, 
            [read_cold_miss_count, write_cold_miss_count], config[0], config[1])

        cur_lat_array = np.multiply(mt_stat_array, exclusive_wb_latency_array)
        cur_mean_lat = np.sum(cur_lat_array)/np.sum(mt_stat_array)

        return mt_stat_array, cur_mean_lat 


    @staticmethod
    def get_mt_stat_from_cumulative_rd_count(cumulative_rd_count, cold_miss_count, tier_1_size, tier_2_size):
        """ Get statistics of a given tier 1 and 2 size for a cumulative RD count and cold miss count. 
        """

        assert tier_1_size+tier_2_size>0, \
            "Tier 1 size is {} and Tier 2 size is {}".format(tier_1_size, tier_2_size)


        """
            The stat array contains the following information. 

            Row 0: tier 1 read hits, tier 2 read hits, read misses 
            Row 1: tier 1 write hits, tier 2 write hits, write misses 
        """
        mt_stat_array = np.zeros([2,3], dtype=int)

        # Read Stats 
        mt_stat_array[0][0] = cumulative_rd_count[tier_1_size-1][0] if tier_1_size > 0 else 0 
        mt_stat_array[0][1] = cumulative_rd_count[tier_1_size+tier_2_size-1][0] \
            - mt_stat_array[0][0] if tier_2_size > 0 else 0 
        mt_stat_array[0][2] = cumulative_rd_count[-1][0] - mt_stat_array[0][0] \
            - mt_stat_array[0][1] + cold_miss_count[0] # Read Misses

        # Write Stats 
        mt_stat_array[1][0] = cumulative_rd_count[tier_1_size-1][1] if tier_1_size > 0 else 0 
        mt_stat_array[1][1] = cumulative_rd_count[tier_1_size+tier_2_size-1][1] \
            - mt_stat_array[1][0] if tier_2_size > 0 else 0  # Tier 2 write hits
        mt_stat_array[1][2] = cumulative_rd_count[-1][1] - mt_stat_array[1][0] \
            - mt_stat_array[1][1] + cold_miss_count[1] # Write Misses 

        return mt_stat_array


    def eval_single_tier(self, cache_size):
        """ Get the read and write hits and misses for a single tier cache. 
        """

        return_data = defaultdict(int)
        for i in range(-1, self.max_rd+1):
            return_data["read_count"] += self.r[i]
            return_data["write_count"] += self.w[i]

            if i>-1 and i<cache_size:
                return_data["read_hit_count"] += self.r[i]
                return_data["write_hit_count"] += self.w[i]
            elif i==-1:
                return_data["read_cold_miss_count"] += self.r[i]
                return_data["write_cold_miss_count"] += self.w[i]
        return return_data 


    def get_read_write_count(self):
        read_count = 0 
        write_count = 0 
        for i in range(-1, self.max_rd+1):
            read_count += self.r[i]
            write_count += self.w[i]
        return read_count, write_count 


    def get_cold_miss_count(self):
        return self.r[-1], self.w[-1]


    def get_hit_rate(self):
        hit_rate_array = []
        read_count, write_count = self.get_read_write_count()

        cur_hit_rate = self.r[-1]/read_count if read_count > 0 else 0.0
        hit_rate_array.append(cur_hit_rate)

        total_read_hit = 0
        for i in range(self.max_rd+1):
            total_read_hit += self.r[i]
            cur_hit_rate = total_read_hit/read_count if read_count > 0 else 0.0
            hit_rate_array.append(cur_hit_rate)

        return hit_rate_array


    def get_cache_size_for_hit_rate(self, target_hit_rate):
        """ Get cache size needed to obtain the specified hit rate. If the specifed 
            hit rate can't be reached, the maximum possible hit rate and the cache size 
            needed to obtain it is returned. 
        """

        read_count, write_count = self.get_read_write_count()
        total_read_hit = 0 
        cache_size = 0
        for i in range(self.max_rd+1):
            total_read_hit += self.r[i]
            
            if read_count == 0:
                current_hit_rate = 0
            else:
                current_hit_rate = total_read_hit/read_count

            if current_hit_rate > target_hit_rate:
                return i+1, total_read_hit/read_count

        if read_count == 0:
            return self.max_rd, 0
        else:
            return self.max_rd, total_read_hit/read_count 


    def get_cache_size_for_normalized_hit_rate(self, normalized_hit_rate):
        """ Get the cache size needed to obtain a normalized hit rate. 
        """

        assert(normalized_hit_rate>0 and normalized_hit_rate<=1)
        read_count, write_count = self.get_read_write_count()
        read_cold_miss_count, write_cold_miss_count = self.get_cold_miss_count()
        total_read_hit = 0
        for i in range(self.max_rd+1):
            total_read_hit += self.r[i]
            cur_hit_rate = total_read_hit/(read_count-read_cold_miss_count) if read_count > 0 else 0.0
            if cur_hit_rate >= normalized_hit_rate:
                return i+1, cur_hit_rate


    @staticmethod 
    def subtract_counter(my_counter, input_counter):
        result_counter = Counter()
        for key in my_counter:
            result_counter[key] = my_counter[key] - input_counter[key]
        return result_counter


    @staticmethod 
    def add_counter(my_counter, input_counter):
        result_counter = Counter()
        for key in my_counter:
            result_counter[key] = my_counter[key] + input_counter[key]
        for key in input_counter:
            if key not in my_counter:
                result_counter[key] = my_counter[key] + input_counter[key]
        return result_counter


    def __add__(self, rd_hist, op="r"):
        new_rd_hist = RDHist()
        new_rd_hist.r = RDHist.add_counter(self.r, rd_hist.r)
        new_rd_hist.w = RDHist.add_counter(self.w, rd_hist.w)
        new_rd_hist.max_rd = max(self.max_rd, rd_hist.max_rd)
        return new_rd_hist


    def __sub__(self, rd_hist):
        new_rd_hist = RDHist()
        new_rd_hist.r = RDHist.subtract_counter(self.r, rd_hist.r)
        new_rd_hist.w = RDHist.subtract_counter(self.w, rd_hist.w)
        new_rd_hist.max_rd = self.max_rd
        return new_rd_hist


    def __eq__(self, other): 
        if not isinstance(other, RDHist):
            return False 

        if self.max_rd == other.max_rd:
            if len(self.r)==len(other.r) and len(self.w)==len(other.w):
                for k in self.r:
                    if self.r[k] != other.r[k]:
                        return False 
                for k in self.w:
                    if self.w[k] != other.w[k]:
                        return False 
            else:
                return False
        else:
            return False 

        return True 

        
