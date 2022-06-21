import argparse 
import pathlib 
from pyCydonia.dataProcessor.cpProcessor import csv_to_page_trace

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate block trace from raw cloudphysics trace")
    parser.add_argument("csv_trace_path", type=pathlib.Path, help="Input path csv")
    parser.add_argument("page_trace_path", type=pathlib.Path, help="Output path of the page trace")
    args = parser.parse_args()

    csv_to_page_trace(args.csv_trace_path, args.page_trace_path, 4096)