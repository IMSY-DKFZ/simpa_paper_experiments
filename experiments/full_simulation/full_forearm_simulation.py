# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import simpa as sp
from simpa import Tags
import numpy as np
import matplotlib.pyplot as plt
from utils.save_directory import get_save_path
from utils.create_example_tissue import create_example_tissue
from utils.basic_settings import create_basic_optical_settings, create_basic_acoustic_settings, \
    create_basic_reconstruction_settings


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VOLUME_WIDTH_HEIGHT_DIM_IN_MM = 50
VOLUME_PLANAR_DIM_IN_MM = 50
SPACING = 0.3
RANDOM_SEED = 24618925

path_manager = sp.PathManager()
SAVE_PATH = get_save_path("full_simulation", "forearm")

# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.


np.random.seed(RANDOM_SEED)
VOLUME_NAME = "ForearmScan" + str(RANDOM_SEED)
file_path = SAVE_PATH + "/" + VOLUME_NAME + ".hdf5"

settings = {
    # These parameters set he general propeties of the simulated volume
    Tags.RANDOM_SEED: RANDOM_SEED,
    Tags.VOLUME_NAME: VOLUME_NAME,
    Tags.SIMULATION_PATH: SAVE_PATH,
    Tags.SPACING_MM: SPACING,
    Tags.WAVELENGTHS: [800],
    Tags.DIM_VOLUME_Z_MM: VOLUME_WIDTH_HEIGHT_DIM_IN_MM,
    Tags.DIM_VOLUME_X_MM: VOLUME_WIDTH_HEIGHT_DIM_IN_MM,
    Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
    Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
    Tags.GPU: True
}

settings = sp.Settings(settings)

settings.set_volume_creation_settings({
    Tags.STRUCTURES: create_example_tissue(),
    Tags.SIMULATE_DEFORMED_LAYERS: True
})

settings.set_optical_settings(create_basic_optical_settings(path_manager))

settings.set_acoustic_settings(create_basic_acoustic_settings(path_manager))

settings.set_reconstruction_settings(create_basic_reconstruction_settings(path_manager, SPACING))

settings["noise_time_series"] = {
    Tags.NOISE_STD: 100,
    Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
    Tags.DATA_FIELD: Tags.DATA_FIELD_TIME_SERIES_DATA
}

device = sp.PhotoacousticDevice(device_position_mm=np.asarray([VOLUME_WIDTH_HEIGHT_DIM_IN_MM/2,
                                                               VOLUME_PLANAR_DIM_IN_MM/2, 0]))
device.set_detection_geometry(sp.LinearArrayDetectionGeometry(device_position_mm=np.asarray([
    VOLUME_WIDTH_HEIGHT_DIM_IN_MM/2, VOLUME_PLANAR_DIM_IN_MM/2, 0]),
                                                              number_detector_elements=256,
                                                              pitch_mm=0.15))
device.add_illumination_geometry(sp.GaussianBeamIlluminationGeometry(beam_radius_mm=25))
SIMUATION_PIPELINE = [
    sp.ModelBasedVolumeCreationAdapter(settings),
    sp.MCXAdapter(settings),
    sp.KWaveAdapter(settings),
    sp.GaussianNoise(settings, "noise_time_series"),
    sp.DelayAndSumAdapter(settings),
    sp.FieldOfViewCropping(settings)
]

sp.simulate(SIMUATION_PIPELINE, settings, device)
wavelength = settings[Tags.WAVELENGTHS][0]

segmentation_mask = sp.load_data_field(file_path=file_path,
                                       wavelength=wavelength,
                                       data_field=Tags.DATA_FIELD_SEGMENTATION)

reco = np.rot90(sp.load_data_field(file_path, wavelength=wavelength, data_field=Tags.DATA_FIELD_RECONSTRUCTED_DATA), -1)
time_series = np.rot90(sp.load_data_field(file_path, wavelength=wavelength, data_field=Tags.DATA_FIELD_TIME_SERIES_DATA), -1)
initial_pressure = np.rot90(sp.load_data_field(file_path, wavelength=wavelength, data_field=Tags.DATA_FIELD_INITIAL_PRESSURE), -1)

plt.figure(figsize=(7, 3))
plt.subplot(1, 3, 1)
plt.axis('off')
plt.imshow(np.fliplr(initial_pressure))
plt.subplot(1, 3, 2)
plt.axis('off')
plt.imshow(np.fliplr(time_series), aspect=0.18)
plt.subplot(1, 3, 3)
plt.axis('off')
plt.imshow(np.fliplr(reco))
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "result.svg"), dpi=300)
plt.close()

# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection='3d')
# ax.voxels(segmentation_mask==SegmentationClasses.EPIDERMIS, shade=True, facecolors="brown", alpha=0.45)
# ax.voxels(segmentation_mask==SegmentationClasses.DERMIS, shade=True, facecolors="pink", alpha=0.45)
# ax.voxels(segmentation_mask==SegmentationClasses.FAT, shade=True, facecolors="yellow", alpha=0.45)
# ax.voxels(segmentation_mask==SegmentationClasses.BLOOD, shade=True, facecolors="red", alpha=0.55)
# ax.voxels(segmentation_mask==SegmentationClasses.BONE, shade=True, facecolors="antiquewhite", alpha=0.55)
# ax.set_aspect('auto')
# ax.set_zlim(50, 0)
# ax.set_zlabel("Depth [mm]")
# ax.set_xlabel("x width [mm]")
# ax.set_ylabel("y width [mm]")
# ax.view_init(elev=10., azim=-45)
# plt.savefig(os.path.join(SAVE_PATH, "forearm.svg"), dpi=300)
# plt.show()