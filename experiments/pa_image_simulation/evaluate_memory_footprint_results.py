from argparse import ArgumentParser
import glob
import os
from simpa.utils import PathManager
from utils.read_log_file import get_times_from_log_file
from utils.read_memory_profile import read_memory_prof_file
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from datetime import datetime, timedelta

parser = ArgumentParser()
parser.add_argument("--spacing_list", default="0.2", type=str)
parser.add_argument("--output_dir", type=str)
config = parser.parse_args()

output_dir = config.output_dir
# if output_dir is None:
#     raise AttributeError("Please specify a directory where the memory footprint files are saved!")
Test = True
if Test:
    output_dir = "/home/kris/Work/Repositories/simpa_paper_experiments/experiments/pa_image_simulation/tmp_memory_footprint"
    config.spacing_list = "0.4 0.35 0.2 0.15"

path_manager = PathManager()
SAVE_PATH = os.path.join(path_manager.get_hdf5_file_save_path(), "Memory_Footprint")

spacing_list = config.spacing_list.split(" ")
spacing_list = [float(i) for i in spacing_list]

output_per_spacing, log_per_spacing = dict(), dict()
for spacing in spacing_list:
    log_file_list = glob.glob(os.path.join(SAVE_PATH, "simpa_{}*.log".format(spacing)))
    log_per_spacing[spacing] = dict()
    output_file_list = glob.glob(os.path.join(output_dir, "mem_spacing_{}*".format(spacing)))
    output_per_spacing[spacing] = dict()

    for run, (log_file, output_file) in enumerate(zip(log_file_list, output_file_list)):
        log_times = get_times_from_log_file(log_file)
        mem_times, memory = read_memory_prof_file(output_file)
        log_start = log_times["start_time"]
        mem_time_start = mem_times[0]
        date_time_time = datetime.fromtimestamp(mem_time_start)
        log_start_time = datetime.strptime(log_start[0] + " " + log_start[1], "%Y-%m-%d %H:%M:%S,%f")
        diff = (log_start_time - date_time_time).total_seconds()
        del log_times["start_time"]
        mem_times -= mem_times[0]
        for key, value in log_times.items():
            log_times[key] = value + diff
        output_per_spacing[spacing][run] = {"times": mem_times, "memory": memory}
        log_per_spacing[spacing][run] = log_times

    plt.figure()
    plt.plot(output_per_spacing[spacing][run]["times"], output_per_spacing[spacing][run]["memory"])
    for key, value in log_per_spacing[spacing][run].items():
        plt.axvline(x=value, label=key, c=np.random.rand(3, ))
    plt.legend()
    plt.show()
    plt.close()

