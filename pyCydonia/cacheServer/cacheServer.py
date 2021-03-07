import math, os 
import numpy as np 
import pandas as pd 
from collections import namedtuple

class CacheServer():

    def __init__(self, tiers):
        self.tiers = tiers


    def get_exclusive_mt_cache_latency_ratio(self) -> np.ndarray:
        """ Returns the relative ratio of latency of different tiers. This is used to 
            penalize the latency incurred by different cache devices. 

            Args:
                self: A CacheServer instance. 

        Returns:
            A [2,3] numpy array where the first index represents read (0) and 
            write(1). The second index represents tier 1 hit latency (0), tier 
            2 hit latency (1), miss latency (2) respectively. For example: 
            
            [[1,2,4],[1,10,10]] where
            The penalty for tier 1 read hit is 1, tier 2 read hit is 2 and a read miss 
            is 4. The penalty of tier 1 write hit is 1, tier 2 and write miss cost the same with 10. 
        """

        cache_latency = self.get_exclusive_wb_mt_cache_latency()
        return np.divide(cache_latency, np.min(cache_latency[np.nonzero(cache_latency)]))


    def get_st_cache_latency_ratio(self, device_index) -> np.ndarray:
        cache_latency = self.get_wb_st_cache_latency(device_index)
        return np.divide(cache_latency, np.min(cache_latency[np.nonzero(cache_latency)]))
    

    def get_wb_st_cache_latency(self, device_index) -> np.ndarray:
        st_wb_latency = np.zeros([2, 3], dtype=float)
        st_wb_latency[0][device_index] = self.tiers[device_index]["read_lat"] # Tier 1 read hit latency 
        st_wb_latency[0][-1] = self.tiers[-1]["read_lat"] + self.tiers[device_index]["write_lat"] 

        st_wb_latency[1][device_index] = self.tiers[device_index]["write_lat"] # Tier 1 write hit latency 
        st_wb_latency[1][-1] = self.tiers[device_index]["write_lat"] # Tier 1 write hit latency 
        return st_wb_latency


    def get_exclusive_wb_mt_cache_latency(self) -> np.ndarray:
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

        exclusive_wb_latency = np.zeros([2, 3], dtype=float)
        exclusive_wb_latency[0][0] = self.tiers[0]["read_lat"] # Tier 1 read hit latency 
        exclusive_wb_latency[0][1] = self.tiers[0]["read_lat"] + self.tiers[1]["read_lat"]  \
            + self.tiers[0]["write_lat"] + self.tiers[1]["write_lat"] # Tier 2 read hit latency 
        exclusive_wb_latency[0][2] = self.tiers[-1]["read_lat"] + self.tiers[0]["write_lat"] \
            + self.tiers[0]["read_lat"] + self.tiers[1]["write_lat"] # Read miss latency 

        exclusive_wb_latency[1][0] = self.tiers[0]["write_lat"] # Tier 1 write hit latency 
        exclusive_wb_latency[1][1] = self.tiers[0]["write_lat"] + self.tiers[0]["read_lat"] \
            + self.tiers[1]["write_lat"] # Tier 2 write hit latency 
        exclusive_wb_latency[1][2] = self.tiers[0]["write_lat"] + self.tiers[0]["read_lat"] \
            + self.tiers[1]["write_lat"] # Write miss latency 

        return exclusive_wb_latency


    def exhaustive_search_opt_exclusive_write_back_mt(self, rd_hist, budget) -> namedtuple:
        """ Find optimal exclusive, write-back MT cache configuration for a budget and workload (rd_hist).

        Args:
            self: A CacheServer instance. 
            rd_hist: A RDHist instance. 
            budget: The max total cost of the MT cache. 

        Returns:
            A namedtuple with fields "min_config" that is a list comprising the size of Tier 1 and Tier 2
            of an optimal multi-tier cache for the budget, "cost_split" that is a list of cost of Tier 1 
            and Tier 2 which add up to less than or equal to the budget and "mean_latency" which is the mean 
            latency of the storage system with the optimal multi tier cache. For example:

            min_config, cost_split, mean_lat = CacheServer.exhaustive_search_opt_exclusive_write_back_mt(rd_hist, budget)
            where 

            min_config = [10,50] where the size of Tier 1 is 10 and Tier 2 is 50. 
            cost_split = [100, 50] where the cost of Tier 1 of size 10 is 100 and Tier 2 of size 50 is 50. 
            mean_latency = 5 where 5 is the mean latency of requests. 
        """

        exclusive_wb_mt_latency = self.get_exclusive_wb_mt_cache_latency()



        mt_cache_hits = rd_hist.get_mt_stat()


    def exhaustive(self, cum_rdhist, miss_count, budget):
        tier1_device = self.tiers[0]
        tier1_max_size = math.floor(budget/tier1_device["price"])
        tier1_max_size = min(tier1_max_size, len(cum_rdhist))

        """
            Get the single-tier data here outside the loop. This is not done inside the loop 
            by ranging from (0, size+1) because there might be remaining budget that wil be 
            allocated to the second tier device making the cache no longer single tier. 
        """
        st_out_wt = self.evaluate_single_tier_cache(cum_rdhist, miss_count, tier1_max_size)
        st_out_wb = self.evaluate_single_tier_cache_wb(cum_rdhist, miss_count, tier1_max_size)
        st_cache_cost = tier1_max_size * tier1_device["price"]

        inclusive_wt_min_entry = [tier1_max_size, 0, st_out_wt[-1]]
        exclusive_wt_min_entry = [tier1_max_size, 0, st_out_wt[-1]]
        inclusive_wb_min_entry = [tier1_max_size, 0, st_out_wb[-1]]
        exclusive_wb_min_entry = [tier1_max_size, 0, st_out_wb[-1]]

        tier2_device = self.tiers[1]
        for tier1_size in range(tier1_max_size):
            rem_budget = budget - tier1_size * tier1_device["price"]
            tier2_size = math.floor(rem_budget/tier2_device["price"])
            tier2_size = min(tier2_size, len(cum_rdhist)-tier1_size)
            total_cache_cost = tier1_size*tier1_device["price"]+tier2_size*tier2_device["price"]

            # evaluate the (tier1_size, tier2_size) combo in different scenarios

            # exclusive write-back 
            exclusive_wb_result = self.evaluate_mt_exclusive_cache_wb(cum_rdhist, 
                miss_count, tier1_size, tier2_size)

            # exclusive write-through 
            exclusive_wt_result = self.evaluate_mt_exclusive_cache(cum_rdhist, 
                miss_count, tier1_size, tier2_size)

            if tier2_size > tier1_size:
                # inclusive write-back 
                inclusive_wt_result = self.evaluate_mt_inclusive_cache(cum_rdhist, 
                    miss_count, tier1_size, tier2_size)

                # inclusive write-through 
                inclusive_wb_result = self.evaluate_mt_inclusive_cache_wb(cum_rdhist, 
                    miss_count, tier1_size, tier2_size)

            if exclusive_wb_min_entry[-1] > exclusive_wb_result[-1]:
                exclusive_wb_min_entry = [tier1_size, tier2_size, exclusive_wb_result[-1]]

            if exclusive_wt_min_entry[-1] > exclusive_wt_result[-1]:
                exclusive_wt_min_entry = [tier1_size, tier2_size, exclusive_wt_result[-1]]

            if inclusive_wb_min_entry[-1] > inclusive_wb_result[-1]:
                inclusive_wb_min_entry = [tier1_size, tier2_size, inclusive_wb_result[-1]]

            if inclusive_wt_min_entry[-1] > inclusive_wt_result[-1]:
                inclusive_wt_min_entry = [tier1_size, tier2_size, inclusive_wt_result[-1]]
            
        return exclusive_wb_min_entry, exclusive_wt_min_entry, inclusive_wb_min_entry, inclusive_wt_min_entry


    def get_inclusive_latency(self):
        assert(len(self.tiers)>2)

        l1_read_lat = self.tiers[0]["read_lat"]
        l2_read_lat = self.tiers[0]["write_lat"] + self.tiers[1]["read_lat"]
        l1_write_lat = self.tiers[0]["write_lat"] + self.tiers[1]["write_lat"] + self.tiers[-1]["write_lat"]
        l2_write_lat = l1_write_lat 
        read_miss_lat = self.tiers[-1]["read_lat"] + self.tiers[0]["write_lat"] + self.tiers[1]["write_lat"]
        write_miss_lat = l1_write_lat
        return [l1_read_lat, l1_write_lat, l2_read_lat, l2_write_lat, read_miss_lat, write_miss_lat]


    def get_inclusive_latency_wb(self):
        assert(len(self.tiers)>2)

        l1_read_lat = self.tiers[0]["read_lat"]
        l2_read_lat = self.tiers[0]["write_lat"] + self.tiers[1]["read_lat"]
        l1_write_lat = self.tiers[0]["write_lat"] + self.tiers[1]["write_lat"] 
        l2_write_lat = l1_write_lat 
        read_miss_lat = self.tiers[-1]["read_lat"] + self.tiers[0]["write_lat"] + self.tiers[1]["write_lat"]
        write_miss_lat = l1_write_lat
        return [l1_read_lat, l1_write_lat, l2_read_lat, l2_write_lat, read_miss_lat, write_miss_lat]


    def get_exclusive_latency(self):
        assert(len(self.tiers)>2)

        l1_read_lat = self.tiers[0]["read_lat"]
        l2_read_lat = self.tiers[0]["read_lat"] + self.tiers[1]["read_lat"] + self.tiers[0]["write_lat"] + self.tiers[1]["write_lat"]
        l1_write_lat = self.tiers[0]["write_lat"] + self.tiers[-1]["write_lat"]
        l2_write_lat = self.tiers[0]["write_lat"] + self.tiers[0]["read_lat"] + self.tiers[1]["write_lat"] + self.tiers[-1]["write_lat"]
        read_miss_lat = self.tiers[-1]["read_lat"] + self.tiers[0]["write_lat"] + self.tiers[0]["read_lat"] + self.tiers[1]["write_lat"]
        write_miss_lat = self.tiers[0]["write_lat"] + self.tiers[0]["read_lat"] + self.tiers[1]["write_lat"] + self.tiers[-1]["write_lat"]
        return [l1_read_lat, l1_write_lat, l2_read_lat, l2_write_lat, read_miss_lat, write_miss_lat]


    def get_exclusive_latency_wb(self):
        assert(len(self.tiers)>2)

        l1_read_lat = self.tiers[0]["read_lat"]
        l1_write_lat = self.tiers[0]["write_lat"] 
        l2_read_lat = self.tiers[0]["read_lat"] + self.tiers[1]["read_lat"] + self.tiers[0]["write_lat"] + self.tiers[1]["write_lat"]
        l2_write_lat = self.tiers[0]["write_lat"] + self.tiers[0]["read_lat"] + self.tiers[1]["write_lat"]
        read_miss_lat = self.tiers[-1]["read_lat"] + self.tiers[0]["write_lat"] + self.tiers[0]["read_lat"] + self.tiers[1]["write_lat"]
        write_miss_lat = self.tiers[0]["write_lat"] + self.tiers[0]["read_lat"] + self.tiers[1]["write_lat"] 
        return [l1_read_lat, l1_write_lat, l2_read_lat, l2_write_lat, read_miss_lat, write_miss_lat]


    def get_single_tier_latency(self):
        l1_read_lat = self.tiers[0]["read_lat"]
        l1_write_lat = self.tiers[0]["write_lat"] + self.tiers[-1]["write_lat"]
        read_miss_lat = self.tiers[-1]["read_lat"] + self.tiers[0]["write_lat"] 
        write_miss_lat = self.tiers[0]["write_lat"] + self.tiers[-1]["write_lat"]
        return [l1_read_lat, l1_write_lat, 0, 0, read_miss_lat, write_miss_lat]


    def get_single_tier_latency_wb(self):
        l1_read_lat = self.tiers[0]["read_lat"]
        l1_write_lat = self.tiers[0]["write_lat"] 
        read_miss_lat = self.tiers[-1]["read_lat"] + self.tiers[0]["write_lat"] 
        write_miss_lat = self.tiers[0]["write_lat"] 
        return [l1_read_lat, l1_write_lat, 0, 0, read_miss_lat, write_miss_lat]


    def get_single_tier_hits_from_rd(self, cum_rdhist, cold_miss_count, l1_size):
        assert(l1_size > 0)

        l1_size = min(l1_size, len(cum_rdhist))
        hit_count = np.array(cum_rdhist[l1_size-1])
        miss_count = np.array(cum_rdhist[-1]) - hit_count
        return [hit_count[0], hit_count[1], 0, 0, miss_count[0]+cold_miss_count[0], miss_count[1]+cold_miss_count[1]]


    def get_inclusive_hits_from_rd(self, cum_rdhist, cold_miss_count, l1_size, l2_size):
        assert(l2_size > l1_size)
        
        l1_size = min(l1_size, len(cum_rdhist))
        l2_size = min(l2_size, len(cum_rdhist))

        l1_hit_count = [0,0]
        if l1_size > 0:
            l1_hit_count = np.array(cum_rdhist[l1_size-1])
        
        l2_hit_count = [0,0]
        if l2_size > 0:
            l2_hit_count = np.array(cum_rdhist[l2_size-1]) - l1_hit_count

        miss_count = cum_rdhist[-1] - l1_hit_count - l2_hit_count
        return [l1_hit_count[0], l1_hit_count[1], l2_hit_count[0], l2_hit_count[1], miss_count[0]+cold_miss_count[0], miss_count[1]+cold_miss_count[1]]


    def get_exclusive_hits_from_rd(self, cum_rdhist, cold_miss_count, l1_size, l2_size):
        l1_size = min(l1_size, len(cum_rdhist))
        l2_size = min(l2_size, len(cum_rdhist)-l1_size)

        l1_hit_count = [0,0]
        if l1_size > 0:
            l1_hit_count = np.array(cum_rdhist[l1_size-1])

        l2_hit_count = [0,0]
        if l2_size > 0:
            l2_hit_count = np.array(cum_rdhist[l1_size+l2_size-1]) - l1_hit_count

        miss_count = cum_rdhist[-1] - l1_hit_count - l2_hit_count
        return [l1_hit_count[0], l1_hit_count[1], l2_hit_count[0], l2_hit_count[1], miss_count[0]+cold_miss_count[0], miss_count[1]+cold_miss_count[1]]


    def evaluate_single_tier_cache(self, cum_rdhist, cold_miss_count, l1_size):
        io_data = self.get_single_tier_hits_from_rd(cum_rdhist, cold_miss_count, l1_size)
        tier_lat = self.get_single_tier_latency()
        mean_latency = np.sum(np.multiply(io_data, tier_lat))/np.sum(io_data)
        output_data = io_data + tier_lat 
        output_data.append(mean_latency)
        return output_data


    def evaluate_single_tier_cache_wb(self, cum_rdhist, cold_miss_count, l1_size):
        io_data = self.get_single_tier_hits_from_rd(cum_rdhist, cold_miss_count, l1_size)
        tier_lat = self.get_single_tier_latency_wb()
        mean_latency = np.sum(np.multiply(io_data, tier_lat))/np.sum(io_data)
        output_data = io_data + tier_lat 
        output_data.append(mean_latency)
        return output_data


    def evaluate_mt_inclusive_cache(self, cum_rdhist, cold_miss_count, l1_size, l2_size):
        io_data = self.get_inclusive_hits_from_rd(cum_rdhist, cold_miss_count, l1_size, l2_size)
        tier_lat = self.get_inclusive_latency()
        mean_latency = np.sum(np.multiply(io_data, tier_lat))/np.sum(io_data)
        output_data = io_data + tier_lat 
        output_data.append(mean_latency)
        return output_data


    def evaluate_mt_inclusive_cache_wb(self, cum_rdhist, cold_miss_count, l1_size, l2_size):
        io_data = self.get_inclusive_hits_from_rd(cum_rdhist, cold_miss_count, l1_size, l2_size)
        tier_lat = self.get_inclusive_latency_wb()
        mean_latency = np.sum(np.multiply(io_data, tier_lat))/np.sum(io_data)
        output_data = io_data + tier_lat 
        output_data.append(mean_latency)
        return output_data


    def evaluate_mt_exclusive_cache(self, cum_rdhist, cold_miss_count, l1_size, l2_size):
        io_data = self.get_exclusive_hits_from_rd(cum_rdhist, cold_miss_count, l1_size, l2_size)
        tier_lat = self.get_exclusive_latency()
        mean_latency = np.sum(np.multiply(io_data, tier_lat))/np.sum(io_data)
        output_data = io_data + tier_lat 
        output_data.append(mean_latency)
        return output_data


    def evaluate_mt_exclusive_cache_wb(self, cum_rdhist, cold_miss_count, l1_size, l2_size):
        io_data = self.get_exclusive_hits_from_rd(cum_rdhist, cold_miss_count, l1_size, l2_size)
        tier_lat = self.get_exclusive_latency_wb()
        mean_latency = np.sum(np.multiply(io_data, tier_lat))/np.sum(io_data)
        output_data = io_data + tier_lat 
        output_data.append(mean_latency)
        return output_data


    def evaluate_cache_of_price(self, cost, l2_device_list, cum_rdhist, cold_miss_count):
        # get the data on single tier cache
        max_l1_size = math.floor((64/65)*(cost/self.tiers[0]["price"]))
        max_l1_size = min(len(cum_rdhist), max_l1_size)
        max_l1_size_mb = math.floor(max_l1_size/256)
        st_out = self.evaluate_single_tier_cache(cum_rdhist, cold_miss_count, max_l1_size)
        st_out_wb = self.evaluate_single_tier_cache_wb(cum_rdhist, cold_miss_count, max_l1_size)

        min_lat_array = np.zeros(shape=(len(l2_device_list),4))
        min_lat_config = np.zeros(shape=(len(l2_device_list), 8))
        min_lat_array.fill(math.inf)

        for cur_l1_size_mb in range(max_l1_size_mb+1):
            print("Evalauting {}/{}".format(cur_l1_size_mb, max_l1_size_mb))
            cur_l1_size = cur_l1_size_mb * 256 
            remaining_cost = cost - (65/64)*(cur_l1_size*self.tiers[0]["price"])
            for l2_device_index, l2_device in enumerate(l2_device_list):
                self.tiers = [self.tiers[0], l2_device, self.tiers[-1]]
                max_l2_size = math.floor((64/(self.tiers[0]["price"]+64*l2_device["price"]))*remaining_cost)    
                total_cost = cur_l1_size*self.tiers[0]["price"] + max_l2_size*l2_device["price"]
                total_cost += self.tiers[0]["price"]*(cur_l1_size+max_l2_size)/64

                if max_l2_size > cur_l1_size:
                    inclusive_out = self.evaluate_mt_inclusive_cache(cum_rdhist, cold_miss_count, cur_l1_size, max_l2_size)
                    if inclusive_out[-1] < min_lat_array[l2_device_index][0]:
                        min_lat_array[l2_device_index][0] = inclusive_out[-1]
                        min_lat_config[l2_device_index][0] = cur_l1_size
                        min_lat_config[l2_device_index][1] = max_l2_size

                    inclusive_out_wb = self.evaluate_mt_inclusive_cache_wb(cum_rdhist, cold_miss_count, cur_l1_size, max_l2_size)
                    if inclusive_out_wb[-1] < min_lat_array[l2_device_index][1]:
                        min_lat_array[l2_device_index][1] = inclusive_out_wb[-1]
                        min_lat_config[l2_device_index][2] = cur_l1_size
                        min_lat_config[l2_device_index][3] = max_l2_size

                exclusive_out = self.evaluate_mt_exclusive_cache(cum_rdhist, cold_miss_count, cur_l1_size, max_l2_size)
                if exclusive_out[-1] < min_lat_array[l2_device_index][2]:
                    min_lat_array[l2_device_index][2] = exclusive_out[-1]
                    min_lat_config[l2_device_index][4] = cur_l1_size
                    min_lat_config[l2_device_index][5] = max_l2_size
                exclusive_out_wb = self.evaluate_mt_exclusive_cache_wb(cum_rdhist, cold_miss_count, cur_l1_size, max_l2_size)
                if exclusive_out_wb[-1] < min_lat_array[l2_device_index][3]:
                    min_lat_array[l2_device_index][3] = exclusive_out_wb[-1]
                    min_lat_config[l2_device_index][6] = cur_l1_size
                    min_lat_config[l2_device_index][7] = max_l2_size

        return min_lat_array, min_lat_config