import pathlib 

""" RDTraceProfiler
    ---------------
    This class generates features related to reuse 
    distance. 

    Parameters
    ----------
    block_reader : CPReader 
        the reader to generate block requests 
    rd_file_path : str 
        path to RD trace file 
"""
class RDTraceProfiler:
    def __init__(self, block_reader, rd_trace_path):
        self._page_size = 4096 
        self._lba_size = 512 
        self._block_reader = block_reader 
        self._rd_trace_path = pathlib.Path(rd_trace_path)


    def get_next_cache_req(self):
        pass 


    def generate_features(self):
        # read one cache request at a time 
        # when there is a new window 
        # track the 
        pass 