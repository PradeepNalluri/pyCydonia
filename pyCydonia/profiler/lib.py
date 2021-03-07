import pathlib 
from pyCydonia.profiler.rdHist import RDHist


def combine_rd_hist_files(rd_hist_path_list):
    """ Given a list of RD histogram files, combine them to return a 
        RDHist object. 
    """

    rd_hist = RDHist()
    num_base_windows_used = 0
    for rd_hist_file_path in rd_hist_path_list:
        if rd_hist_file_path != 0:
            rd_hist.update_rd_hist_file(str(rd_hist_file_path))
            num_base_windows_used += 1
    return rd_hist, num_base_windows_used


def get_rd_hist_file_list_for_window(rd_hist_dir, workload_name, start_index, reconfig_window_size):
    """ For an entry in window details csv, get the list of relevant RD histogram files. 
    """

    rd_hist_file_list = []
    rd_hist_dir = pathlib.Path(rd_hist_dir)
    for cur_index in range(start_index, start_index + reconfig_window_size):
        rd_hist_file_name = "{}_{}.csv".format(workload_name, cur_index)
        rd_hist_file_path = rd_hist_dir.joinpath(rd_hist_file_name)
        if rd_hist_file_path.is_file():
            rd_hist_file_list.append(str(rd_hist_file_path))
    return rd_hist_file_list 