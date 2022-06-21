import argparse 
import pathlib 
from pyCydonia.dataProcessor.cpProcessor import raw_trace_to_page_trace

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate block trace from raw cloudphysics trace")
    parser.add_argument("raw_trace_path", type=pathlib.Path, help="Input raw trace")
    parser.add_argument("page_trace_path", type=pathlib.Path, help="Output page trace")
    args = parser.parse_args()

    raw_trace_to_page_trace(args.raw_trace_path, args.page_trace_path, 4096)