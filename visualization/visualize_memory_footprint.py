from simpa.visualisation.matplotlib_data_visualisation import visualise_data
from simpa.utils.path_manager import PathManager
import os

path_manager = PathManager()
SAVE_PATH = path_manager.get_hdf5_file_save_path()
SAVE_PATH = os.path.join(SAVE_PATH, "Memory_Footprint")
spacings = [0.3]
# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.
for spacing in spacings:
    VOLUME_NAME = "Spacing_{}_Seed_{}".format(spacing, 500)
    visualise_data(800, os.path.join(SAVE_PATH, VOLUME_NAME) + ".hdf5",
                   show_absorption=True,
                   show_initial_pressure=True,
                   show_reconstructed_data=True,
                   log_scale=False)