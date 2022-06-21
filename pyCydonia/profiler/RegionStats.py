

class RegionStats:
    def __init__(self):
        pass 


    def _region_access_list(self, req, region_size):
        """ This function finds the regions accessed 
            by the current block request. The size of 
            the region is defined by the user. 

            Parameters
            ----------
            req : object 
                an object containing block request features
            region_size : int 
                the size of a region in bytes 

            Return
            ------
            it : iterator 
                an iterator that returns each region accessed 
        """

        start_offset = self.req_start_offset(req)
        start_region = start_offset//region_size 
        end_offset = self.req_end_offset(req)
        end_region = (end_offset-1)//region_size 
        return range(start_region, end_region+1)


    def _update_region_access(self, req, index, region_size):
        """ This function updates the counter of number of accesses 
            to regions of different sizes. 

            Parameters
            ----------
            req : object 
                an object containing block request features
            index : int
                the index of the counter in self._counter_array
            region_size : int 
                the size of a region in bytes 
        """

        for region_index in self._region_access_list(req, region_size):
            self._counter_array[index][region_index] += 1


    def _track_region_access(self, req):
        """ Track the regions of different sizes listed in 
            self._region_size_array that are being accessed 
            by the current requests. 

            Parameters
            ----------
            req : object 
                an object containing block request features
        """

        for index, region_size in enumerate(self._region_size_array):
            self._update_region_access(req, index, region_size)