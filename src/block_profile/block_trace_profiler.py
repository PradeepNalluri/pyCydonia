import argparse 
import pathlib 
from pyCydonia.reader.CPReader import CPReader
from pyCydonia.profiler.BlockTraceProfiler import BlockTraceProfiler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                "Profile a given block trace and generate features"
            )

    parser.add_argument(
                "block_trace_path", 
                type=pathlib.Path, 
                help="Path to the block trace in CSV format")
    
    parser.add_argument(
                "--rd", 
                default=None,
                type=pathlib.Path, 
                help="Path to trace with reuse distance of each page request")

    parser.add_argument(
                "--out", 
                default=None, 
                type=pathlib.Path, 
                help="Output path for stats (stdout if nothing passed)")

    args = parser.parse_args()
    reader = CPReader(args.block_trace_path)

    if args.rd is None:
        profiler = BlockTraceProfiler(reader, 
                        ["block"],
                        window_size=30*60*1e6,
                        rd_dir="/research/file_system_traces/cp_traces/mtcache/rd_traces_4k",
                        page_dir="/research/file_system_traces/cp_traces/mtcache/page_traces_4k",
                        rd_snapshot_dir="/research/file_system_traces/cp_traces/mtcache/rd_snapshot_4k",
                        workload_snapshot_dir="/research/file_system_traces/cp_traces/mtcache/workload_snapshot_4k")
        profiler.run()