import argparse 
import pathlib 
from pyCydonia.profiler.nprdHist import NPHist

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot MRC from a RD histogram file")
    parser.add_argument("rd_hist_path", type=pathlib.Path, help="Rd Histogram file path")
    parser.add_argument("mrc_plot_path", type=pathlib.Path, help="Output path of the MRC plot")
    args = parser.parse_args()

    rd_hist = NPHist()
    rd_hist.load(args.rd_hist_path)
    rd_hist.plot_read_mrc(args.mrc_plot_path, max_mrc_size_mb=8*1024)

    