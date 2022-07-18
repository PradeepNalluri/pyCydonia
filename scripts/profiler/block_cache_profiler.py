import argparse
from genericpath import exists 
import pathlib

from pyCydonia.dataProcessor.cpProcessor import raw_trace_to_block_trace
from pyCydonia.reader.CPReader import CPReader
from pyCydonia.profiler.BlockTraceProfiler import BlockTraceProfiler

import multiprocessing as mp
from joblib import Parallel, delayed
import copy
import pandas as pd
import numpy as np

NUM_CORES = int(mp.cpu_count())

def worker_function(main_args):
    """
    Worker function for the parallelization.
    @param args: Object for sending in the arguments to the worker function in parallel.
    """
    for args in main_args:
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
        del profiler
        del reader
        del block_snapshot_dir
    return 0
if __name__ == "__main__":
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

    parser.add_argument(
        "--limit_file_size",
        type=int,
        default=-1,
        help="""Limit the size of the file to be process, default: -1(meaning no limit). Else the limit is in MB.)"""
    )

    parser.add_argument(
        "--backend",
        default="multiprocessing",
        choices=['multiprocessing', 'joblib'],
        help="""Type of multiprocessing strategy to use. defualt: multiprocessing. Accepted args: multiprocessing, joblib."""
    )

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
        done_files = pd.read_csv(args.out_dir.joinpath("block.csv"))['workload'].tolist()
        for file in files:
            args.block_trace_path = pathlib.Path(file)
            if(args.block_trace_path.name.replace('.csv','') in done_files):
                continue
            limit_file_size = args.limit_file_size
            if(limit_file_size==-1 or args.block_trace_path.stat().st_size<limit_file_size*1024*1024):
                contracts.append(copy.deepcopy(args))
        print("Total number of contracts: ", len(contracts))
        contracts = np.split(np.array(contracts), num_jobs)
        use_mp = args.backend=="multiprocessing"

        # create a pool of workers to run the feature extraction in parallel
        if(use_mp):
            pool = mp.Pool(num_jobs)
            results = pool.map(worker_function,contracts)
            pool.close()
            pool.join()
        else:
            results = Parallel(n_jobs=num_jobs, verbose=10)(delayed(worker_function)(contract) for contract in contracts)

    else:
        raise ValueError("Invalid mode")
