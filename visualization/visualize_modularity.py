# SPDX-FileCopyrightText: 2021 Intelligent Medical Systems Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa.visualisation.matplotlib_data_visualisation import visualise_data
from simpa.utils.path_manager import PathManager
import os

path_manager = PathManager()
SAVE_PATH = path_manager.get_hdf5_file_save_path()
SAVE_PATH = os.path.join(SAVE_PATH, "Modularity_Examples")

devices = ["InVision256TF", "MSOTAcuityEcho"]
recon_algorithms = ["TR", "DAS"]
# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.
for device_key in devices:
    for recon_algorithm_key in recon_algorithms:
        VOLUME_NAME = "Device_{}_Recon_{}_Seed_{}".format(device_key, recon_algorithm_key, 500)
        visualise_data(800, os.path.join(SAVE_PATH, VOLUME_NAME) + ".hdf5",
                       show_absorption=False,
                       show_initial_pressure=True,
                       show_DATA_FIELD_RECONSTRUCTED_DATA=True,
                       log_scale=False)