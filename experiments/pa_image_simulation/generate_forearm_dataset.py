"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa import Tags
import matplotlib.pyplot as plt
import numpy as np
import simpa as sp
from utils.create_example_tissue import create_forearm_dataset_tissue
from matplotlib_scalebar.scalebar import ScaleBar
from utils.normalizations import normalize_min_max
from utils.save_directory import get_save_path
from utils.basic_settings import create_basic_optical_settings, create_basic_acoustic_settings, \
    create_basic_reconstruction_settings
from mpl_toolkits.axes_grid1 import ImageGrid
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
path_manager = sp.PathManager()
SAVE_PATH = get_save_path("pa_image_simulation", "Forearm_Dataset")

SPACING = 0.15
RANDOM_SEED = 1234
WAVELENGTHS = [700]
SHOW_IMAGE = False
dataset_size = 16

start_time = time.time()
for run in range(dataset_size):
    RANDOM_SEED += 1
    np.random.seed(RANDOM_SEED)
    settings = sp.Settings()
    settings[Tags.SIMULATION_PATH] = SAVE_PATH
    settings[Tags.VOLUME_NAME] = f"Forearm_image_{run + 1}"
    settings[Tags.RANDOM_SEED] = RANDOM_SEED
    settings[Tags.WAVELENGTHS] = WAVELENGTHS
    settings[Tags.SPACING_MM] = SPACING
    settings[Tags.DIM_VOLUME_Z_MM] = 20
    settings[Tags.DIM_VOLUME_X_MM] = 40
    settings[Tags.DIM_VOLUME_Y_MM] = 20
    settings[Tags.GPU] = True

    settings.set_optical_settings(create_basic_optical_settings(path_manager))

    settings.set_acoustic_settings(create_basic_acoustic_settings(path_manager))

    settings.set_reconstruction_settings(create_basic_reconstruction_settings(path_manager, SPACING))
    settings[Tags.RECONSTRUCTION_MODEL_SETTINGS][Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION] = True
    settings[Tags.RECONSTRUCTION_MODEL_SETTINGS][Tags.RECONSTRUCTION_MODE] = Tags.RECONSTRUCTION_MODE_DIFFERENTIAL

    settings["noise_initial_pressure"] = {
        Tags.NOISE_MEAN: 1,
        Tags.NOISE_STD: 0.05,
        Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
        Tags.DATA_FIELD: Tags.DATA_FIELD_INITIAL_PRESSURE,
        Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
    }

    settings["noise_time_series"] = {
        Tags.NOISE_STD: 25,
        Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
        Tags.DATA_FIELD: Tags.DATA_FIELD_TIME_SERIES_DATA
    }

    sp.create_deformation_settings(
        bounds_mm=[[0, settings[Tags.DIM_VOLUME_X_MM]],
                   [0, settings[Tags.DIM_VOLUME_Y_MM]]],
        maximum_z_elevation_mm=4,
        filter_sigma=0.1,
        cosine_scaling_factor=1.5)

    settings.set_volume_creation_settings({
        Tags.STRUCTURES: create_forearm_dataset_tissue(settings),
        Tags.SIMULATE_DEFORMED_LAYERS: True,
        Tags.US_GEL: True
    })

    device = sp.MSOTAcuityEcho(
        device_position_mm=np.array([40/2, 20/2, -3]),
        field_of_view_extent_mm=np.asarray([-20, 20, 0, 0, 0, 20]))
    device.update_settings_for_use_of_model_based_volume_creator(settings)

    SIMUATION_PIPELINE = [
        sp.ModelBasedVolumeCreationAdapter(settings),
        sp.MCXAdapter(settings),
        sp.GaussianNoise(settings, "noise_initial_pressure"),
        sp.KWaveAdapter(settings),
        sp.GaussianNoise(settings, "noise_time_series"),
        sp.DelayAndSumAdapter(settings),
        sp.FieldOfViewCropping(settings)
    ]

    sp.simulate(SIMUATION_PIPELINE, settings, device)

print("####################\nThe Simulations took {}s\n####################".format(time.time() - start_time))

fig = plt.figure(figsize=(10, 8))
img_grid = ImageGrid(fig, rect=111, nrows_ncols=(3, 4), axes_pad=0.05)

for run, gridax in enumerate(img_grid):
    print(run)
    recon = sp.load_data_field(SAVE_PATH + "/" + f"Forearm_image_{run + 1}" + ".hdf5",
                               Tags.DATA_FIELD_RECONSTRUCTED_DATA, wavelength=WAVELENGTHS[0])

    recon = normalize_min_max(recon)
    img_to_plot = np.rot90(recon, 3)
    img_plot = gridax.imshow(img_to_plot)
    if run == 8:
        scale_bar = ScaleBar(SPACING, units="mm", location="lower left",
                             font_properties={"family": "Cmr10", "size": 10})
        gridax.add_artist(scale_bar)
    gridax.axis("off")
if SHOW_IMAGE:
    plt.show()
    plt.close()
else:
    plt.savefig(SAVE_PATH + "/dataset_grid.svg")
    plt.close()
