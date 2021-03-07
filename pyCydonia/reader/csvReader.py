# coding=utf-8

"""
Reader for CP traces.  

TODO: Currently reader assumes that time is in microseconds

Author: Pranav Bhandari <bhandaripranav94@gmail.com> 2020/11
"""

import pathlib 
from pyCydonia.reader.reader import Reader


class CSVReader(Reader):
    
    def __init__(self, file_path, reader_config):
        super(CSVReader, self).__init__(file_path)
        self.file_path = file_path 
        self.file_handler = open(file_path, "r")
        self.__config = reader_config 


    def get_num_lines(self):
        """ The function returns the number of lines in a file.
        """

        line_count = 0 
        with open(self.file_path) as f:
            line = f.readline().rstrip()
            while line:
                line_count += 1
                line = f.readline().rstrip()
        return line_count 


    def get_next_line(self):
        """ The function returns the next line on the file. 
        """

        return self.file_handler.readline().rstrip()


    def get_next_line_json(self):
        """ Get the data of the next line in JSON. 
        """

        line = self.get_next_line()
        if not line:
            return None 
        return self.line_to_json(line)


    def get_data_between_time_range(self, start_time, end_time):
        """ The function returns the block request between the start and end time specified. The time is 
            in seconds and is relative to the start of the trace. For instance, start time of 0 means the 
            beginning of the trace, start time of 300 means starting from 5 minutes into the trace. If the 
            trace ends before the end of the specified interval, whatever data was collected up until that 
            point of time is returned. 
        """

        temp_reader = CSVReader(self.file_path, self.__config)
        line_json = temp_reader.get_next_line_json()

        trace_start_time = int(int(line_json["time"])/1000000)
        
        cur_line_index = 0
        cur_window_data = []
        while line_json:

            trace_current_time = int(int(line_json["time"])/1000000)
            cur_relative_time = trace_current_time - trace_start_time
            
            if cur_relative_time >= end_time:
                break
            elif cur_relative_time >= start_time and cur_relative_time < end_time:
                line_json["index"] = cur_line_index
                cur_window_data.append(line_json)

            line_json = temp_reader.get_next_line_json()
            cur_line_index += 1

        return cur_window_data 


    def get_next_window(self, window_size_sec, reset_reader=True):
        """ Get the portion of the block trace file corresponding to the next time window. 
        """

        if reset_reader:
            self.reset()

        cur_window_index = 0 
        cur_window_data = []
        line_json = self.get_next_line_json()
        trace_start_time = int(line_json["time"])
        while line_json:
            current_time = int(line_json["time"])

            # convert window in seconds into microseconds and get current time window index 
            cur_time_window_index = int((current_time - trace_start_time)/(1000000*int(window_size_sec)))
            while cur_time_window_index > cur_window_index:
                yield cur_window_data 
                cur_window_data = []
                cur_window_index += 1 

            cur_window_data.append(line_json)
            line_json = self.get_next_line_json()
        yield cur_window_data


    def generate_split_trace(self, window_size_sec, output_dir, workload_name="w"):
        """ Generate split block trace corresponding to specified window size. 
        """

        window_index = 0 
        for window in self.get_next_window(window_size_sec):
            window_file_name = "{}_{}.csv".format(workload_name, window_index)
            window_file_path = pathlib.Path(output_dir).joinpath(window_file_name)
            with window_file_path.open("w+") as f:
                for line_json in window:
                    line_string = "{},{},{}\n".format(line_json["lba"], line_json["op"], 
                        line_json["time"])
                    f.write(line_string)
            window_index += 1


    def line_to_json(self, line):
        """ Convert the line to JSON based on the configuration. 
        """

        split_line = line.split(self.__config["delimiter"])
        line_json = {}
        for key in self.__config["data"]:
            line_json[key] = split_line[int(self.__config["data"][key]["index"])]
        return line_json 


    def reset(self):
        """ Reset the file descriptor offset to 0. 
        """

        self.file_handler.seek(0)


    def skip_lines(self, num_lines):
        """ Skip n lines of the file before reading a line. Used for ommitting headers that could yield error
            when being read. 
        """

        for _ in range(num_lines):
            self.get_next_line()

            
