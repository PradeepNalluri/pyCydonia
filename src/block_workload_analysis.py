import pandas as pd 

df = pd.read_csv("block_trace_stat.csv", names = [
        "trace_name",
        "min_offset",
        "range_mb",
        "io_size_mb",
        "read_io_ratio",
        "io_req_rate",
        "mean_read_size",
        "mean_write_size",
        "time_len_days"
    ])


# sort based on io_size_mb 

print(df.sort_values(["io_size_mb", "read_io_ratio"]))