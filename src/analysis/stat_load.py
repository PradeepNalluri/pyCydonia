import pandas as pd 

header_list = [
    "trace_name",
    "min_offset",
    "range_mb",
    "io_size_mb",
    "read_io_ratio",
    "mean_io_req_rate",
    "max_io_req_rate",
    "max_read_io_req_rate",
    "max_write_io_req_rate",
    "mean_read_size",
    "mean_write_size",
    "time_len_days",
]

df = pd.read_csv("../block_trace_stat.csv", names=header_list)

print(df["mean_read_size"].describe())
print(df["mean_write_size"].describe())


print(df[df["read_io_ratio"]>0.75])
print(df.sort_values(["io_size_mb", "read_io_ratio"], ascending=(True, True)))

