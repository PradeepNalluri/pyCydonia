import argparse
from genericpath import exists 
import pathlib 

from pyCydonia.reader.CPReader import CPReader
from pyCydonia.profiler.BlockTraceProfiler import BlockTraceProfiler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                "Profile a given block trace and generate block cache features."
            )

    parser.add_argument(
                "block_trace_path", 
                type=pathlib.Path, 
                help="Path to the block trace in CSV format")

    parser.add_argument(
                "--out_dir", 
                default=pathlib.Path("/research/file_system_traces/cp_traces/BlockTraceFeatures/cloudphysics"), 
                type=pathlib.Path, 
                help="Output dir for stats")

    args = parser.parse_args()

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