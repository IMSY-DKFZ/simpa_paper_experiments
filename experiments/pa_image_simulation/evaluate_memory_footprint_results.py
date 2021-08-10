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
parser.add_argument("--plot_spacing", default=0.2, type=float)
parser.add_argument("--output_dir", type=str)
config = parser.parse_args()

output_dir = config.output_dir
# if output_dir is None:
#     raise AttributeError("Please specify a directory where the memory footprint files are saved!")
Test = True
if Test:
    output_dir = "/home/kris/Work/Repositories/simpa_paper_experiments/experiments/pa_image_simulation/tmp_memory_footprint"
    config.spacing_list = "0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41"


color_list = [""]
path_manager = PathManager()
SAVE_PATH = os.path.join(path_manager.get_hdf5_file_save_path(), "Memory_Footprint")

spacing_list = config.spacing_list.split(" ")
spacing_list = [float(i) for i in spacing_list]

output_per_spacing, log_per_spacing = dict(), dict()
max_ram_per_module, time_per_module = dict(), dict()
modules = ["vol_creation_start", "opt_sim_start", "ac_sim_start", "rec_start", "crop_start"]
for module in modules:
    max_ram_per_module[module] = dict()
    time_per_module[module] = dict()

for sp, spacing in enumerate(spacing_list):
    log_file_list = sorted(glob.glob(os.path.join(SAVE_PATH, "simpa_{}.log".format(spacing))))
    log_per_spacing[spacing] = dict()
    output_file_list = sorted(glob.glob(os.path.join(output_dir, "mem_spacing_{}-run*".format(spacing))))
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

    previous_stop = 0
    next_key = "Initializing"
    for key, value in log_per_spacing[spacing][run].items():
        tt = output_per_spacing[spacing][run]["times"]
        stop = value
        previous_stop = previous_stop
        points_in_gate = (previous_stop <= tt) & (tt <= stop)
        time_in_gate = tt[points_in_gate]
        ram_in_gate = output_per_spacing[spacing][run]["memory"][points_in_gate]
        if next_key != "Initializing":
            max_ram_per_module[next_key][spacing] = max(ram_in_gate)
            time_per_module[next_key][spacing] = time_in_gate[-1] - time_in_gate[0]

        if spacing == config.plot_spacing:
            plt.plot(time_in_gate, ram_in_gate, label=next_key)
            plt.axvline(x=value, c="black", ls=":")
        previous_stop = value
        next_key = key

    ### Now for the ending time ###
    if spacing == config.plot_spacing:
        tt = output_per_spacing[spacing][run]["times"]
        previous_stop = previous_stop
        points_in_gate = (previous_stop <= tt)
        plt.plot(tt[points_in_gate],
                 output_per_spacing[spacing][run]["memory"][points_in_gate], label="Terminating")

        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("RAM Usage [MB]")
        plt.show()
        plt.close()
        # exit()

for module in modules:
    lists = sorted(max_ram_per_module[module].items())
    x, y = zip(*lists)

    plt.plot(x, y, label=module)
ax = plt.gca()
ax.invert_xaxis()
plt.legend()
plt.xlabel("Spacing [mm]")
plt.ylabel("RAM Usage [MB]")
plt.semilogy()
plt.show()

for module in modules:
    lists = sorted(time_per_module[module].items())
    x, y = zip(*lists)

    plt.plot(x, y, label=module)
ax = plt.gca()
ax.invert_xaxis()
plt.legend()
plt.xlabel("Spacing [mm]")
plt.ylabel("Runtime [s]")
plt.semilogy()
plt.show()
