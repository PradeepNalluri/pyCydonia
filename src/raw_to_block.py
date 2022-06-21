import argparse 
import pathlib 
from pyCydonia.dataProcessor.cpProcessor import raw_trace_to_block_trace

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate block trace from raw cloudphysics trace")
    parser.add_argument("raw_trace_path", type=pathlib.Path, help="Input path of the raw trace")
    parser.add_argument("block_trace_path", type=pathlib.Path, help="Output path of the block trace file")
    args = parser.parse_args()

    raw_trace_to_block_trace(args.raw_trace_path, args.block_trace_path)