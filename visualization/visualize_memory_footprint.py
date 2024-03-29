# SPDX-FileCopyrightText: 2021 Intelligent Medical Systems Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import simpa as sp
import os

path_manager = sp.PathManager()
SAVE_PATH = path_manager.get_hdf5_file_save_path()
SAVE_PATH = os.path.join(SAVE_PATH, "Memory_Footprint")
spacings = [0.3]
# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.
for spacing in spacings:
    VOLUME_NAME = "Spacing_{}_Seed_{}".format(spacing, 500)
    sp.visualise_data(800, os.path.join(SAVE_PATH, VOLUME_NAME) + ".hdf5",
                      show_absorption=True,
                      show_initial_pressure=True,
                      log_scale=False)
