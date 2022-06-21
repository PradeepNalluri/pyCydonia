import argparse 
import pathlib 
from pyCydonia.dataProcessor.cpProcessor import raw_trace_to_page_trace


def all_raw_trace_to_page_trace(raw_trace_dir, csv_trace_dir, page_size=4096):
    for raw_trace_path in raw_trace_dir.iterdir():
        if any([char.isdigit() for char in raw_trace_path.stem]):
            workload_name = raw_trace_path.stem.split("_")[0]
            csv_trace_path = csv_trace_dir.joinpath("{}.csv".format(workload_name))
            raw_trace_to_page_trace(raw_trace_path, csv_trace_path, page_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate CSV cache traces from raw cloudphysics traces")
    parser.add_argument("raw_trace_dir", type=pathlib.Path, help="Input path of the raw trace")
    parser.add_argument("cache_trace_dir", type=pathlib.Path, help="Output path of the block trace file")
    args = parser.parse_args()

    all_raw_trace_to_page_trace(args.raw_trace_dir, args.cache_trace_dir)