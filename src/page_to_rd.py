import argparse 
import pathlib 
from pyCydonia.dataProcessor.cpProcessor import page_file_to_rd

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate block trace from raw cloudphysics trace")
    parser.add_argument("page_trace_path", type=pathlib.Path, help="Path to the page trace")
    parser.add_argument("rd_trace_path", type=pathlib.Path, help="Output path of the block trace file")
    args = parser.parse_args()

    page_file_to_rd(str(args.page_trace_path), str(args.rd_trace_path))