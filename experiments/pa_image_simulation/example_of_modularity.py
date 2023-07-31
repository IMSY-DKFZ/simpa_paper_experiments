# SPDX-FileCopyrightText: 2021 Intelligent Medical Systems Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import matplotlib.pyplot as plt
import numpy as np
import simpa as sp
from simpa import Tags
import time
import os
from utils.create_example_tissue import create_square_phantom
from utils.basic_settings import create_basic_reconstruction_settings, create_basic_optical_settings, \
    create_basic_acoustic_settings
from simpa.visualisation.matplotlib_data_visualisation import visualise_data
from utils.save_directory import get_save_path
from matplotlib_scalebar.scalebar import ScaleBar
from shutil import copy
from visualization.colorbar import col_bar
from utils.normalizations import normalize_min_max


VOLUME_TRANSDUCER_DIM_IN_MM = 90
VOLUME_PLANAR_DIM_IN_MM = 20
VOLUME_HEIGHT_IN_MM = 90
SPACING = 0.15
RANDOM_SEED = 500
# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = sp.PathManager()

WAVELENGTHS = [800]
SAVE_PATH = get_save_path("pa_image_simulation", "Modularity_Examples")

devices = {"MSOTAcuityEcho": sp.MSOTAcuityEcho, "InVision256TF": sp.InVision256TF}
recon_algorithms = {"TR": sp.TimeReversalAdapter, "DAS": sp.DelayAndSumAdapter}
# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.
np.random.seed(RANDOM_SEED)

results = dict()
start_time = time.time()
for dv, (device_key, device) in enumerate(devices.items()):
    results[device_key] = dict()

    VOLUME_NAME = "Device_{}_Seed_{}".format(device_key, RANDOM_SEED)

    general_settings = {
        # These parameters set the general properties of the simulated volume
        Tags.RANDOM_SEED: RANDOM_SEED,
        Tags.VOLUME_NAME: VOLUME_NAME,
        Tags.SIMULATION_PATH: SAVE_PATH,
        Tags.SPACING_MM: SPACING,
        Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
        Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
        Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
        Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
        Tags.GPU: True,
        Tags.WAVELENGTHS: WAVELENGTHS,
    }
    settings = sp.Settings(general_settings)

    settings.set_volume_creation_settings({
        Tags.STRUCTURES: create_square_phantom(settings),
        Tags.US_GEL: True,

    })

    settings.set_optical_settings(create_basic_optical_settings(path_manager))

    settings.set_acoustic_settings(create_basic_acoustic_settings(path_manager))

    if device_key == "MSOTAcuityEcho":
        pa_device = device(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                        VOLUME_PLANAR_DIM_IN_MM / 2,
                                                        25]),
                           field_of_view_extent_mm=np.asarray([-20, 20,
                                                               0, 0, 0, 40])
                           )
        pa_device.update_settings_for_use_of_model_based_volume_creator(settings)
    else:
        pa_device = device(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                        VOLUME_PLANAR_DIM_IN_MM / 2,
                                                        VOLUME_HEIGHT_IN_MM / 2]))

    SIMUATION_PIPELINE = [
        sp.ModelBasedVolumeCreationAdapter(settings),
        sp.MCXAdapter(settings),
        sp.KWaveAdapter(settings),
        sp.FieldOfViewCropping(settings)
    ]

    sp.simulate(SIMUATION_PIPELINE, settings, pa_device)
    p0 = sp.load_data_field(SAVE_PATH + "/" + VOLUME_NAME + ".hdf5",
                            Tags.DATA_FIELD_INITIAL_PRESSURE, WAVELENGTHS[0])
    p0 = normalize_min_max(p0)
    results[device_key]["Initial Pressure"] = np.rot90(p0, -1)

    for recon_algorithm_key, recon_algorithm in recon_algorithms.items():

        VOLUME_NAME_RECON = "Device_{}_Recon_{}_Seed_{}".format(device_key, recon_algorithm_key, RANDOM_SEED)
        copy(os.path.join(SAVE_PATH, VOLUME_NAME) + ".hdf5", os.path.join(SAVE_PATH, VOLUME_NAME_RECON) + ".hdf5")

        settings[Tags.VOLUME_NAME] = VOLUME_NAME_RECON
        settings[Tags.SIMPA_OUTPUT_PATH] = os.path.join(SAVE_PATH, VOLUME_NAME_RECON) + ".hdf5"

        settings.set_reconstruction_settings(
            create_basic_reconstruction_settings(path_manager, reconstruction_spacing=settings[Tags.SPACING_MM]))

        recon_module = recon_algorithm(settings)
        recon_module.run(pa_device)
        cropping_module = sp.FieldOfViewCropping(settings)
        cropping_module.run(pa_device)

        # Load simulated results #
        file = sp.load_hdf5(SAVE_PATH + "/" + VOLUME_NAME_RECON + ".hdf5")
        recon = sp.load_data_field(SAVE_PATH + "/" + VOLUME_NAME_RECON + ".hdf5",
                                Tags.DATA_FIELD_RECONSTRUCTED_DATA, WAVELENGTHS[0])
        recon = normalize_min_max(recon)
        results[device_key][recon_algorithm_key] = np.rot90(recon, -1)
end_time = time.time() - start_time
with open(os.path.join(SAVE_PATH, "run_time.txt"), "w+") as out_file:
    out_file.write("{:.2f} s".format(end_time))

plt.figure(figsize=(10, 6))
for i, (device, results) in enumerate(results.items()):
    for r, image in enumerate(["Initial Pressure", "TR", "DAS"]):
        plt.subplot(2, 3, i*3 + r + 1)
        plt.title(image)
        if r == 0:
            plt.ylabel(device)
        img_plot = plt.imshow(results[image])
        scale_bar = ScaleBar(settings[Tags.SPACING_MM], units="mm")
        plt.gca().add_artist(scale_bar)
        col_bar(img_plot)

SHOW_IMAGE = False
if SHOW_IMAGE:
    plt.show()
else:
    plt.savefig(SAVE_PATH + "/Modularity_Example.svg")
