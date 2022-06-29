import argparse
from genericpath import exists 
import pathlib 

from pyCydonia.dataProcessor.cpProcessor import raw_trace_to_block_trace
from pyCydonia.reader.CPReader import CPReader
from pyCydonia.profiler.BlockTraceProfiler import BlockTraceProfiler

import multiprocessing as mp
import copy

NUM_CORES = int(mp.cpu_count())

def worker_function(args):
    """
    Worker function for the parallelization.
    @param args: Object for sending in the arguments to the worker function in parallel.
    """
    # create a directory for this workload 
    workload_name = args.block_trace_path.stem 
    workload_dir = args.out_dir.joinpath(workload_name)
    workload_dir.mkdir(exist_ok=True)

    # create directory for block stat snapshots 
    block_snapshot_dir = workload_dir.joinpath("block_snapshots")
    block_snapshot_dir.mkdir(exist_ok=True)

    reader = CPReader(args.block_trace_path)
    profiler = BlockTraceProfiler(reader, 
                    ["block"],
                    window_size=30*60*1e6, # 30 minutes 
                    snapshot_dir=block_snapshot_dir)
    profiler.generate_features(out_path=args.out_dir.joinpath("block.csv"))

if __name__ == "__main__":
    traceFile = "w83_vscsi2.vscsitrace"
    csvFile = "w83.csv"
    #raw_trace_to_block_trace("/net/nap1/workloads/mt-caching/traces/cloudphysics/raw/"+traceFile, pathlib.Path("/home/lglee/pyCydonia/workloads/"+csvFile)) 
   # print("done processing csv")
    #exit()
    parser = argparse.ArgumentParser(
                "Profile a given block trace and generate block cache features."
            )

    parser.add_argument(
                "block_trace_path", 
                type=pathlib.Path, 
                help="""Path to the block trace in CSV format, 
                    if the mode is multi file mode then path to
                    the directory containing the block traces.""")

    parser.add_argument(
                "--out_dir", 
                default=pathlib.Path("/research/file_system_traces/cp_traces/BlockTraceFeatures/cloudphysics"),
                type=pathlib.Path, 
                help="Output dir for stats")
    
    parser.add_argument(
                "--mode",
                type=int,
                default=0,
                help="0: single file mode, 1: multiple file mode")

    parser.add_argument(
                "--num_jobs",
                type=int,
                default=-1,
                help="""Number of jobs to run in parallel, -1 to use all cores. default: -1.
                "Only used in multiple file mode.""")

    args = parser.parse_args()

    if(args.mode==0):
        worker_function(args)

    elif(args.mode==1):
        num_jobs = args.num_jobs if args.num_jobs>0 else NUM_CORES
        # get all the csv files in the directory
        files = list(args.block_trace_path.glob("w*.csv"))
        
        if(len(files)==0):
            print("No workload files found in the directory")
            exit(1)

        contracts = []
        for file in files:
            args.block_trace_path = pathlib.Path(file)
            contracts.append(copy.deepcopy(args))

        # create a pool of workers to run the feature extraction in parallel
        pool = mp.Pool(NUM_CORES)
        results = pool.map(worker_function,contracts)
        pool.close()
        pool.join()

    else:
        raise ValueError("Invalid mode")
