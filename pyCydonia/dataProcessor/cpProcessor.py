import pathlib 
import math 

from PyMimircache import Cachecow
from pyCydonia.profiler.rdHist import RDHist


DEFAULT_PAGE_TRACE_CONFIG = {
    "init_params": {
            "label": 1, # the indexing start from 1 
            "real_time": 3, 
            "op": 2, 
            "delimiter": ","
    },
    "label": 1,
    "delimiter": ","
}


def rd_file_to_rd_hist(rd_file):
    """ Generate a RD Hist object from a rd file. 
    """

    rd_hist = RDHist()
    rd_hist.load_rd_file(str(rd_file))
    return rd_hist 


def rd_file_to_rd_hist_file(rd_file, rd_hist_file):
    """ Given a rd file, generate a rd hist file. 
    """

    print("rd_file_to_rd_hist_file({}, {})".format(rd_file,
        rd_hist_file))

    rd_hist = rd_file_to_rd_hist(rd_file)
    rd_hist.write_to_file(str(rd_hist_file))


def sanity_check_rd_hist(rd_file, rd_hist_file):
    """ Tests if the rd_file and the rd_hist_file generates the same RD Histogram object. 

    """

    print("sanity_check_rd_hist({}, {})".format(rd_file,
        rd_hist_file))

    rd_hist_from_rd_file = RDHist()
    rd_hist_from_rd_file.load_rd_file(str(rd_file))

    rd_hist_from_rd_hist_file = RDHist()
    rd_hist_from_rd_hist_file.load_rd_hist_file(str(rd_hist_file))

    assert rd_hist_from_rd_file==rd_hist_from_rd_hist_file, \
        "The reuse distance histograms generated from the RD file {} and the RD Hist file" \
        " {} is not the same!".format(rd_file, rd_hist_file)

    print("Sanity Check Passed!")


def sanity_check_rd_file(page_trace_file, rd_file):
    """ Performs sanity check of a page trace file and its subsequent RD file. It ensures 
        that the operation of each line is the same and that the file contains the same 
        number of lines. 

    """

    print("sanity_check_rd_file({}, {})".format(page_trace_file,
        rd_file))

    if type(page_trace_file) is not pathlib.Path:
        page_trace_file = pathlib.Path(page_trace_file)

    if type(rd_file) is not pathlib.Path:
        rd_file = pathlib.Path(rd_file)

    with page_trace_file.open("r") as page_handle:
        with rd_file.open("r") as rd_handle:
            page_trace_line = page_handle.readline().rstrip()
            rd_line = rd_handle.readline().rstrip()
            line_count = 0 
            while page_trace_line:
                
                page_op = page_trace_line.split(",")[1]
                rd_op = rd_line.split(",")[1]

                assert page_op == rd_op, \
                    "The operation at line {} does not match, \n Line 1: {} \n Line 2: {}\n".format(line_count,
                        page_trace_line, rd_line)

                page_trace_line = page_handle.readline().rstrip()
                rd_line = rd_handle.readline().rstrip()
                line_count += 1

            assert page_trace_line == rd_line, \
                "The end of files not same at line {}. \n File 1: {}\n File 2: {}\n".format(line_count, 
                    page_trace_line, rd_trace_line)
    print("Sanity Check Passed!")


def page_file_to_rd(page_file_path, rd_file_path):
    """ Generates an RD file for a page file.
    """

    print("page_file_to_rd({}, {})".format(page_file_path,
        rd_file_path))

    rd = get_rd_from_page_trace(page_file_path)
    with open (str(page_file_path), "r") as page_file_handle:
        with open(str(rd_file_path), "w+") as rd_file_handle:
            for i in range(len(rd)):
                rd_file_handle.write("{},{}\n".format(rd[i],
                    page_file_handle.readline().rstrip().split(",")[1]))


def setup_mimircache_for_raw_trace(vscsi_file_path, vscsi_type):
    """ Return a Mimircache for the correct VSCSI type. 

    """

    assert(vscsi_type==1 or vscsi_type==2)
    mimircache = Cachecow()
    mimircache.vscsi(vscsi_file_path, vscsi_type=vscsi_type)
    return mimircache


def get_rd_from_page_trace(page_file_path, reader_params=DEFAULT_PAGE_TRACE_CONFIG):
    """ Generate a RD array from a page accesses file. 
    """

    mimircache = Cachecow()
    reader = mimircache.csv(str(page_file_path), reader_params)
    profiler = mimircache.profiler(algorithm="LRU")
    return profiler.get_reuse_distance()


def raw_trace_to_page_trace(raw_trace_file_path, page_trace_file_path, page_size, lba_size=512):
    """ Concert raw VSCSI files to a page access file. 
    """

    print("raw_trace_to_access_trace({}, {})".format(raw_trace_file_path,
        page_trace_file_path))

    # Read and Write Operation Codes for this trace 
    READ_OP_CODES = [40,8,168,136,9]
    WRITE_OP_CODES = [42,10,170,138,11]

    raw_trace_file_path = pathlib.Path(raw_trace_file_path)
    assert raw_trace_file_path.is_file(), \
        "{} passed as raw trace is not a file!".format(raw_trace_file_path)

    raw_trace_file_name = raw_trace_file_path.name
    vscsi_type = 2 if "vscsi2" in raw_trace_file_name else 1 

    # setup reader and write file handle before starting to read the I/O requests
    mimircache = setup_mimircache_for_raw_trace(str(raw_trace_file_path), vscsi_type)
    reader = mimircache.reader
    write_file_handle = page_trace_file_path.open("w+")

    """
        Read an io which is an array of components of the request. 
        It differs based on vscsi type. 
        For eg: ["Size", "LBA", "R/W"]
    """
    io = reader.read_complete_req() 
    print("Sample IO: {}".format(io))

    while io is not None:

        # index of size and op_code (VSCSI op_code) are different in type 1 and 2 
        """
            VSCSI Type 1: X, size, X, op code, X, LBA, time milliseconds 
            VSCSI Type 2: op_code, X, 
        """
        if vscsi_type == 1:
            size = int(io[1])
            op_code = int(io[3])
        else:
            size = int(io[3])
            op_code = int(io[0])

        lba = io[5]
        time_ms = int(io[6])

        num_lba_accessed = math.ceil(size/lba_size)

        """
            Refer to the VSCSI manual. The problem is that the op code 127 can be for 
            both read and write. Therefore, if we find an entry with that op code in a
            trace, we require the user to handle that and make it clear weather it was a
            read or a write by manually changing it. 
        """
        if op_code == 127:
            print("READ/WRITE CONFUSION io_type 127!")
            sys.exit()

        if op_code in READ_OP_CODES:

            # based on VSCSI manual, if op_code is 8 and size if 0 that actually means read the next 256 blocks 
            if size == 0 and op_code == 8:
                num_lba_accessed = 256
            io_type = "r"
        elif op_code in WRITE_OP_CODES:

            # based on VSCSI manual, if op_code is 10 and size if 0 that actually means write the next 256 blocks 
            if size == 0 and op_code == 10:
                num_lba_accessed = 256
            io_type = "w"
        else:

            # ignore the op codes that do not belong to read and write 
            io = reader.read_complete_req()
            continue

        # we know read/write and the number of blocks so we can output that to block file 
        for i in range(num_lba_accessed):
            cur_page = int((lba*lba_size)/page_size)
            write_file_handle.write("{},{},{}\n".format(cur_page, io_type, time_ms))
            lba += 1 

        io = reader.read_complete_req()
    
    write_file_handle.close()
