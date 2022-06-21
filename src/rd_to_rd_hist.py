import argparse 
import pathlib 
from pyCydonia.dataProcessor.cpProcessor import rd_file_to_rd_hist_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate block trace from raw cloudphysics trace")
    parser.add_argument("rd_trace_path", type=pathlib.Path, help="Output path of the block trace file")
    parser.add_argument("rd_hist_path", type=pathlib.Path, help="Output path of the block trace file")
    args = parser.parse_args()

    rd_file_to_rd_hist_file(str(args.rd_trace_path), str(args.rd_hist_path))