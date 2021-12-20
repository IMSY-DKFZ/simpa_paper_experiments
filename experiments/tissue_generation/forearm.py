# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa import Tags
import simpa as sp
import numpy as np
import matplotlib.pyplot as plt
from simpa.utils.libraries.structure_library import define_horizontal_layer_structure_settings, \
    define_vessel_structure_settings, define_circular_tubular_structure_settings, define_background_structure_settings
from utils.save_directory import get_save_path


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VOLUME_WIDTH_HEIGHT_DIM_IN_MM = 50
VOLUME_PLANAR_DIM_IN_MM = 50
SPACING = 0.5
RANDOM_SEED = 24618925

path_manager = sp.PathManager()
SAVE_PATH = get_save_path("Tissue_Generation", "Forearm")


def create_example_tissue():
    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = define_background_structure_settings(molecular_composition=sp.TISSUE_LIBRARY.ultrasound_gel())
    tissue_dict["epidermis"] = define_horizontal_layer_structure_settings(z_start_mm=0, thickness_mm=2,
                                                                          adhere_to_deformation=True,
                                                                          molecular_composition=sp.TISSUE_LIBRARY.epidermis())
    tissue_dict["dermis"] = define_horizontal_layer_structure_settings(z_start_mm=2, thickness_mm=9,
                                                                       adhere_to_deformation=True,
                                                                       molecular_composition=sp.TISSUE_LIBRARY.dermis())
    tissue_dict["fat"] = define_horizontal_layer_structure_settings(z_start_mm=11, thickness_mm=4,
                                                                    adhere_to_deformation=True,
                                                                    molecular_composition=sp.TISSUE_LIBRARY.subcutaneous_fat())
    tissue_dict["vessel_1"] = define_vessel_structure_settings(vessel_start_mm=[25, 0, 17],
                                                               vessel_direction_mm=[-0.05, 1, 0],
                                                               radius_mm=2, bifurcation_length_mm=100,
                                                               curvature_factor=0.01,
                                                               molecular_composition=sp.TISSUE_LIBRARY.blood())
    tissue_dict["vessel_2"] = define_vessel_structure_settings(vessel_start_mm=[5, 0, 17],
                                                               vessel_direction_mm=[0, 1, 0],
                                                               radius_mm=1.5, bifurcation_length_mm=100,
                                                               curvature_factor=0.01,
                                                               molecular_composition=sp.TISSUE_LIBRARY.blood())
    tissue_dict["vessel_3"] = define_vessel_structure_settings(vessel_start_mm=[45, 0, 19],
                                                               vessel_direction_mm=[0.05, 1, 0],
                                                               radius_mm=1.5, bifurcation_length_mm=100,
                                                               curvature_factor=0.01,
                                                               molecular_composition=sp.TISSUE_LIBRARY.blood())
    tissue_dict["vessel_4"] = define_vessel_structure_settings(vessel_start_mm=[25, 0, 35],
                                                               vessel_direction_mm=[0.05, 1, 0],
                                                               radius_mm=6, bifurcation_length_mm=15,
                                                               curvature_factor=0.1,
                                                               molecular_composition=sp.TISSUE_LIBRARY.blood())
    tissue_dict["bone"] = define_circular_tubular_structure_settings(tube_start_mm=[5, 0, 45], tube_end_mm=[5, 50, 45],
                                                                     radius_mm=15,
                                                                     molecular_composition=sp.TISSUE_LIBRARY.bone())
    return tissue_dict

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
    Tags.WAVELENGTHS: [700],
    Tags.DIM_VOLUME_Z_MM: VOLUME_WIDTH_HEIGHT_DIM_IN_MM,
    Tags.DIM_VOLUME_X_MM: VOLUME_WIDTH_HEIGHT_DIM_IN_MM,
    Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
    Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE
}

settings = sp.Settings(settings)

settings.set_volume_creation_settings({
    Tags.STRUCTURES: create_example_tissue(),
    Tags.SIMULATE_DEFORMED_LAYERS: True
})


device = sp.RSOMExplorerP50()
SIMUATION_PIPELINE = [
    sp.ModelBasedVolumeCreationAdapter(settings)
]
import time
start_time = time.time()
sp.simulate(SIMUATION_PIPELINE, settings, device)
end_time = time.time() - start_time
with open(os.path.join(SAVE_PATH, "run_time.txt"), "w+") as out_file:
    out_file.write("{:.2f} s".format(end_time))
wavelength = settings[Tags.WAVELENGTHS][0]

segmentation_mask = sp.load_data_field(file_path=file_path,
                                       wavelength=wavelength,
                                       data_field=Tags.DATA_FIELD_SEGMENTATION)
fontsize = 13
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.voxels(segmentation_mask==sp.SegmentationClasses.EPIDERMIS, shade=True, facecolors="brown", alpha=0.45)
ax.voxels(segmentation_mask==sp.SegmentationClasses.DERMIS, shade=True, facecolors="pink", alpha=0.45)
ax.voxels(segmentation_mask==sp.SegmentationClasses.FAT, shade=True, facecolors="yellow", alpha=0.45)
ax.voxels(segmentation_mask==sp.SegmentationClasses.BLOOD, shade=True, facecolors="red", alpha=0.55)
ax.voxels(segmentation_mask==sp.SegmentationClasses.BONE, shade=True, facecolors="antiquewhite", alpha=0.55)
ax.set_aspect('auto')
# ax.set_xticks(np.linspace(0, settings[Tags.DIM_VOLUME_X_MM]/settings[Tags.SPACING_MM], 6))
# ax.set_yticks(np.linspace(0, settings[Tags.DIM_VOLUME_Y_MM]/settings[Tags.SPACING_MM], 6))
# ax.set_zticks(np.linspace(0, settings[Tags.DIM_VOLUME_Z_MM]/settings[Tags.SPACING_MM], 6))
# ax.set_xticklabels(np.linspace(0, settings[Tags.DIM_VOLUME_X_MM], 6, dtype=int), fontsize=fontsize)
# ax.set_yticklabels(np.linspace(0, settings[Tags.DIM_VOLUME_X_MM], 6, dtype=int), fontsize=fontsize)
# ax.set_zticklabels(np.linspace(0, settings[Tags.DIM_VOLUME_X_MM], 6, dtype=int), fontsize=fontsize)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_zlim(int(settings[Tags.DIM_VOLUME_X_MM]/settings[Tags.SPACING_MM]), 0)
# ax.set_zlabel("Depth [mm]", fontsize=fontsize)
# ax.set_xlabel("x width [mm]", fontsize=fontsize)
# ax.set_ylabel("y width [mm]", fontsize=fontsize)
# plt.axis("off")
ax.view_init(elev=10., azim=-45)
plt.savefig(os.path.join(SAVE_PATH, "forearm.svg"), dpi=300)
plt.close()