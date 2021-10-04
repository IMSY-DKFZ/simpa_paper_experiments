from argparse import ArgumentParser
import glob
import os
import numpy as np
from simpa.utils import PathManager
from utils.read_log_file import get_times_from_log_file
from utils.read_memory_profile import read_memory_prof_file
import matplotlib.pyplot as plt
from datetime import datetime
from utils.save_directory import get_save_path
plt.style.use("bmh")

parser = ArgumentParser()
parser.add_argument("--spacing_list", default="0.2", type=str)
parser.add_argument("--plot_spacing", default=0.2, type=float)
parser.add_argument("--output_dir", type=str)
config = parser.parse_args()

SHOW_IMAGE = False

output_dir = config.output_dir
# if output_dir is None:
#     raise AttributeError("Please specify a directory where the memory footprint files are saved!")
SAVE_PATH = get_save_path("pa_image_simulation", "Memory_Footprint")
Test = True
if Test:
    output_dir = os.path.join(SAVE_PATH, "ram_usage_logs")
    config.spacing_list = "0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4"
    # config.spacing_list = "0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4"

spacing_list = config.spacing_list.split(" ")
spacing_list = [float(i) for i in spacing_list if i != ""]

output_per_spacing, log_per_spacing = dict(), dict()
max_ram_per_module, time_per_module = dict(), dict()
module_names = {"vol_creation_start": "Volume creation",
                "opt_sim_start": "Optical simulation",
                "ac_sim_start": "Acoustic simulation",
                "rec_start": "Image reconstruction",
                "crop_start": "Field of view cropping"}
modules = module_names.keys()
for module in modules:
    max_ram_per_module[module] = dict()
    time_per_module[module] = dict()
    for spacing in spacing_list:
        max_ram_per_module[module][spacing] = dict()
        time_per_module[module][spacing] = dict()
plt.figure(figsize=(15, 7))
for sp, spacing in enumerate(spacing_list):
    log_file_list = sorted(glob.glob(os.path.join(SAVE_PATH, "simpa_spacing_{}_run_*.log".format(spacing))))
    log_per_spacing[spacing] = dict()
    output_file_list = sorted(glob.glob(os.path.join(output_dir, "mem_spacing_{}-run*".format(spacing))))
    output_per_spacing[spacing] = dict()
    number_of_runs = len(output_file_list)

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

    for run in range(number_of_runs):
        previous_stop = 0
        next_key = "Initializing"
        for i, (key, value) in enumerate(log_per_spacing[spacing][run].items()):
            tt = output_per_spacing[spacing][run]["times"]
            stop = value
            previous_stop = previous_stop
            points_in_gate = (previous_stop <= tt) & (tt <= stop)
            time_in_gate = tt[points_in_gate]
            ram_in_gate = output_per_spacing[spacing][run]["memory"][points_in_gate]
            if next_key != "Initializing":
                max_ram_per_module[next_key][spacing][run] = max(ram_in_gate)
                time_per_module[next_key][spacing][run] = time_in_gate[-1] - time_in_gate[0]

            if spacing == config.plot_spacing and run == 0:
                if next_key in modules:
                    label = module_names[next_key]
                else:
                    label = next_key
                plt.plot(time_in_gate, np.array(ram_in_gate)/1000, label=label)
                plt.fill_between(time_in_gate, np.array(ram_in_gate)/1000,
                                 alpha=0.5,
                                 linewidth=2)
                plt.axvline(x=value, c="black", ls=":", alpha=0.5)
            previous_stop = value
            next_key = key

        ### Now for the ending time ###
        if spacing == config.plot_spacing and run == 0:
            tt = output_per_spacing[spacing][run]["times"]
            previous_stop = previous_stop
            points_in_gate = (previous_stop <= tt)
            plt.plot(tt[points_in_gate],
                     np.array(output_per_spacing[spacing][run]["memory"][points_in_gate])/1000, label="Terminating")
            plt.fill_between(tt[points_in_gate],
                             np.array(output_per_spacing[spacing][run]["memory"][points_in_gate])/1000,
                             alpha=0.5,
                             linewidth=2)

            plt.legend(loc="best")
            plt.xlabel("Time [s]", fontsize=15)
            plt.ylabel("RAM Usage [GB]", fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            if SHOW_IMAGE:
                plt.show()
            else:
                plt.savefig(os.path.join(SAVE_PATH, "Example_RAM_curve.svg"))
            plt.close()
            # exit()
plt.figure(figsize=(15, 7))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for m, module in enumerate(modules):
    std_ram_list = list()
    for spacing in spacing_list:
        run_list = list()
        for run in range(number_of_runs):
            run_list.append(max_ram_per_module[module][spacing][run])
        mean_ram = np.mean(run_list)
        std_ram = np.std(run_list)
        max_ram_per_module[module][spacing] = mean_ram
        std_ram_list.append(std_ram)

    max_lists = sorted(max_ram_per_module[module].items())
    x, y = zip(*max_lists)

    plt.plot(x, np.array(y) / 1000, label=module_names[module], color=colors[m + 1], linewidth=5)
    plt.fill_between(x, (np.array(y) - np.array(std_ram_list))/1000, (np.array(y) + np.array(std_ram_list))/1000, color=colors[m + 1], alpha=0.5)
ax = plt.gca()
ax.invert_xaxis()
ax.set_facecolor("white")
plt.legend(prop={"size":20})
plt.xlabel("Spacing [mm]", fontsize=15)
plt.ylabel("Peak RAM Usage [GB]", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.semilogy()
if SHOW_IMAGE:
    plt.show()
else:
    plt.savefig(os.path.join(SAVE_PATH, "RAM_scaling.svg"))
plt.close()

plt.figure(figsize=(15, 7))

for m, module in enumerate(modules):
    std_time_list = list()
    for spacing in spacing_list:
        run_list = list()
        for run in range(number_of_runs):
            run_list.append(time_per_module[module][spacing][run])
        mean_time = np.mean(run_list)
        std_time = np.std(run_list)
        time_per_module[module][spacing] = mean_time
        std_time_list.append(std_time)

    max_lists = sorted(time_per_module[module].items())
    x, y = zip(*max_lists)
    plt.plot(x, y, label=module_names[module], color=colors[m + 1], linewidth=5)
    plt.fill_between(x, np.array(y) - np.array(std_time_list), np.array(y) + np.array(std_time_list), color=colors[m + 1], alpha=0.5)

ax = plt.gca()
ax.invert_xaxis()
ax.set_facecolor("white")
plt.legend(prop={"size": 20})
plt.xlabel("Spacing [mm]", fontsize=15)
plt.ylabel("Runtime [s]", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.semilogy()
if SHOW_IMAGE:
    plt.show()
else:
    plt.savefig(os.path.join(SAVE_PATH, "Time_scaling.svg"))
plt.close()
