import argparse 
import pathlib 
from pyCydonia.dataProcessor.cpProcessor import raw_trace_to_block_trace


def all_raw_trace_to_csv_trace(raw_trace_dir, csv_trace_dir):
    for raw_trace_path in raw_trace_dir.iterdir():
        if any([char.isdigit() for char in raw_trace_path.stem]):
            workload_name = raw_trace_path.stem.split("_")[0]
            csv_trace_path = csv_trace_dir.joinpath("{}.csv".format(workload_name))
            raw_trace_to_block_trace(raw_trace_path, csv_trace_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate CSV block traces from raw cloudphysics traces")
    parser.add_argument("cloudphysics_dir", type=pathlib.Path, help="Input path of the raw trace")
    parser.add_argument("csv_trace_dir", type=pathlib.Path, help="Output path of the block trace file")
    args = parser.parse_args()

    all_raw_trace_to_csv_trace(args.cloudphysics_dir, args.csv_trace_dir)