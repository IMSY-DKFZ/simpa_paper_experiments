# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa import Tags
import simpa as sp
import numpy as np
import matplotlib.pyplot as plt
from utils.save_directory import get_save_path
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VOLUME_WIDTH_HEIGHT_DIM_IN_MM = 50
VOLUME_PLANAR_DIM_IN_MM = 50
SPACING = 0.5
RANDOM_SEED = 2736587

path_manager = sp.PathManager()

SAVE_PATH = get_save_path("Tissue_Generation", "Phantom")
VOLUME_NAME = "PhantomScan" + str(RANDOM_SEED)
file_path = SAVE_PATH + "/" + VOLUME_NAME + ".hdf5"


def create_example_tissue():
    """
    This is a very simple example script of how to create a tissue definition.
    It contains a muscular background, an epidermis layer on top of the muscles
    and a blood vessel.
    """
    background_dictionary = sp.Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.ultrasound_gel()
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    phantom_material_dictionary = sp.Settings()
    phantom_material_dictionary[Tags.PRIORITY] = 3
    phantom_material_dictionary[Tags.STRUCTURE_START_MM] = [VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2,
                                                            0,
                                                            VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2]
    phantom_material_dictionary[Tags.STRUCTURE_END_MM] = [VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2,
                                                          VOLUME_PLANAR_DIM_IN_MM,
                                                          VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2]
    phantom_material_dictionary[Tags.STRUCTURE_RADIUS_MM] = 14
    phantom_material_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.soft_tissue()
    phantom_material_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = False
    phantom_material_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    inclusion_1_dictionary = sp.Settings()
    inclusion_1_dictionary[Tags.PRIORITY] = 5
    inclusion_1_dictionary[Tags.STRUCTURE_START_MM] = [VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2 + 7 * np.sin(np.deg2rad(10)),
                                                       0,
                                                       VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2 + 7 * np.cos(np.deg2rad(10))]
    inclusion_1_dictionary[Tags.STRUCTURE_END_MM] = [VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2 + 7 * np.sin(np.deg2rad(10)),
                                                     VOLUME_PLANAR_DIM_IN_MM,
                                                     VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2 + 7 * np.cos(np.deg2rad(10))]
    inclusion_1_dictionary[Tags.STRUCTURE_RADIUS_MM] = 2
    inclusion_1_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood()
    inclusion_1_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = False
    inclusion_1_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    inclusion_2_dictionary = sp.Settings()
    inclusion_2_dictionary[Tags.PRIORITY] = 5
    inclusion_2_dictionary[Tags.STRUCTURE_START_MM] = [VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2 - 7 * np.sin(np.deg2rad(10)),
                                                       0,
                                                       VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2 - 7 * np.cos(np.deg2rad(10))]
    inclusion_2_dictionary[Tags.STRUCTURE_END_MM] = [VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2 - 7 * np.sin(np.deg2rad(10)),
                                                     VOLUME_PLANAR_DIM_IN_MM,
                                                     VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2 - 7 * np.cos(np.deg2rad(10))]
    inclusion_2_dictionary[Tags.STRUCTURE_RADIUS_MM] = 2
    inclusion_2_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood()
    inclusion_2_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = False
    inclusion_2_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    tissue_dict["phantom"] = phantom_material_dictionary
    tissue_dict["inclusion_1"] = inclusion_1_dictionary
    tissue_dict["inclusion_2"] = inclusion_2_dictionary
    return tissue_dict

# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.


np.random.seed(RANDOM_SEED)

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
    Tags.SIMULATE_DEFORMED_LAYERS: False
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
ax.voxels(segmentation_mask == sp.SegmentationClasses.BLOOD, shade=True, facecolors="red", alpha=0.55)
ax.voxels(segmentation_mask == sp.SegmentationClasses.MUSCLE, shade=True, facecolors="yellow", alpha=0.15)
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
ax.view_init(elev=10., azim=-45)
plt.savefig(os.path.join(SAVE_PATH, "phantom.svg"), dpi=300)
plt.close()
