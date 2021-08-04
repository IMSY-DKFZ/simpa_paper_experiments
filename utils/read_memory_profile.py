import pandas as pd
import os
import glob


def read_memory_prof_file(path):
    df = pd.read_csv(path, sep=" ", skiprows=1, header=None, usecols=[1, 2])
    df = df.to_numpy()
    memory_mb = df[:, 0]
    time_s = df[:, 1]

    return time_s, memory_mb


def read_most_recent_memory_prof_files(directory, number_of_files: int = 1):
    memory_prof_file_list = glob.glob(os.path.join(directory, "memory_profile*.csv"))
    sorted_memory_prof_file_list = sorted(memory_prof_file_list)
    time_list, memory_list = list(), list()
    for file in sorted_memory_prof_file_list[:number_of_files]:
        times, memories = read_memory_prof_file(file)
        time_list.append(times)
        memory_list.append(memories)

    return time_list, memory_list


if __name__ == "__main__":
    from simpa.utils.path_manager import PathManager
    path_manager = PathManager()
    memory_profile_folder = os.path.join(path_manager.get_hdf5_file_save_path(), "Memory_Footprint")
    times, mems = read_most_recent_memory_prof_files(memory_profile_folder)
    print(times)

