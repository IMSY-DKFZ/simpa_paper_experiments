from simpa.utils import Tags, SegmentationClasses
from simpa.utils.path_manager import PathManager

from simpa.core.simulation import simulate
from simpa.core import VolumeCreationModelModelBasedAdapter
from simpa.utils import Settings, TISSUE_LIBRARY
from simpa.core.device_digital_twins import RSOMExplorerP50
from simpa.io_handling import load_data_field
import numpy as np
import inspect
import matplotlib.pyplot as plt
from simpa.utils.libraries.structure_library import define_horizontal_layer_structure_settings, \
    define_vessel_structure_settings, define_circular_tubular_structure_settings, define_background_structure_settings


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VOLUME_WIDTH_HEIGHT_DIM_IN_MM = 50
VOLUME_PLANAR_DIM_IN_MM = 50
SPACING = 1.0
RANDOM_SEED = 24618925

path_manager = PathManager()

base_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def create_example_tissue():
    tissue_dict = Settings()
    tissue_dict[Tags.BACKGROUND] = define_background_structure_settings(molecular_composition=TISSUE_LIBRARY.ultrasound_gel())
    tissue_dict["epidermis"] = define_horizontal_layer_structure_settings(z_start_mm=0, thickness_mm=2,
                                                                          adhere_to_deformation=True,
                                                                          molecular_composition=TISSUE_LIBRARY.epidermis())
    tissue_dict["dermis"] = define_horizontal_layer_structure_settings(z_start_mm=2, thickness_mm=9,
                                                                          adhere_to_deformation=True,
                                                                       molecular_composition=TISSUE_LIBRARY.dermis())
    tissue_dict["fat"] = define_horizontal_layer_structure_settings(z_start_mm=11, thickness_mm=4,
                                                                       adhere_to_deformation=True,
                                                                       molecular_composition=TISSUE_LIBRARY.subcutaneous_fat())
    tissue_dict["vessel_1"] = define_vessel_structure_settings(vessel_start_mm=[25, 0, 17],
                                                               vessel_direction_mm=[-0.05, 1, 0],
                                                               radius_mm=2, bifurcation_length_mm=100,
                                                               curvature_factor=0.01,
                                                               molecular_composition=TISSUE_LIBRARY.blood())
    tissue_dict["vessel_2"] = define_vessel_structure_settings(vessel_start_mm=[5, 0, 17],
                                                               vessel_direction_mm=[0, 1, 0],
                                                               radius_mm=1.5, bifurcation_length_mm=100,
                                                               curvature_factor=0.01,
                                                               molecular_composition=TISSUE_LIBRARY.blood())
    tissue_dict["vessel_3"] = define_vessel_structure_settings(vessel_start_mm=[45, 0, 19],
                                                               vessel_direction_mm=[0.05, 1, 0],
                                                               radius_mm=1.5, bifurcation_length_mm=100,
                                                               curvature_factor=0.01,
                                                               molecular_composition=TISSUE_LIBRARY.blood())
    tissue_dict["vessel_4"] = define_vessel_structure_settings(vessel_start_mm=[25, 0, 35],
                                                               vessel_direction_mm=[0.05, 1, 0],
                                                               radius_mm=6, bifurcation_length_mm=15,
                                                               curvature_factor=0.1,
                                                               molecular_composition=TISSUE_LIBRARY.blood())
    tissue_dict["bone"] = define_circular_tubular_structure_settings(tube_start_mm=[5, 0, 45], tube_end_mm=[5, 50, 45],
                                                                     radius_mm=15,
                                                                     molecular_composition=TISSUE_LIBRARY.bone())
    return tissue_dict

# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.


np.random.seed(RANDOM_SEED)

settings = {
    # These parameters set he general propeties of the simulated volume
    Tags.RANDOM_SEED: RANDOM_SEED,
    Tags.VOLUME_NAME: "ForearmScan" + str(RANDOM_SEED),
    Tags.SIMULATION_PATH: "./",
    Tags.SPACING_MM: SPACING,
    Tags.WAVELENGTHS: [700],
    Tags.DIM_VOLUME_Z_MM: VOLUME_WIDTH_HEIGHT_DIM_IN_MM,
    Tags.DIM_VOLUME_X_MM: VOLUME_WIDTH_HEIGHT_DIM_IN_MM,
    Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
    Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE
}

settings = Settings(settings)

settings.set_volume_creation_settings({
    Tags.STRUCTURES: create_example_tissue(),
    Tags.SIMULATE_DEFORMED_LAYERS: True
})


device = RSOMExplorerP50()
SIMUATION_PIPELINE = [
    VolumeCreationModelModelBasedAdapter(settings)
]

simulate(SIMUATION_PIPELINE, settings, device)
wavelength = settings[Tags.WAVELENGTHS][0]

segmentation_mask = load_data_field(file_path=settings[Tags.VOLUME_NAME] + ".hdf5",
                                    wavelength=wavelength,
                                    data_field=Tags.PROPERTY_SEGMENTATION)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.voxels(segmentation_mask==SegmentationClasses.EPIDERMIS, shade=True, facecolors="brown", alpha=0.45)
ax.voxels(segmentation_mask==SegmentationClasses.DERMIS, shade=True, facecolors="pink", alpha=0.45)
ax.voxels(segmentation_mask==SegmentationClasses.FAT, shade=True, facecolors="yellow", alpha=0.45)
ax.voxels(segmentation_mask==SegmentationClasses.BLOOD, shade=True, facecolors="red", alpha=0.55)
ax.voxels(segmentation_mask==SegmentationClasses.BONE, shade=True, facecolors="antiquewhite", alpha=0.55)
ax.set_aspect('auto')
ax.set_zlim(50, 0)
ax.set_zlabel("Depth [mm]")
ax.set_xlabel("x width [mm]")
ax.set_ylabel("y width [mm]")
ax.view_init(elev=10., azim=-45)
plt.savefig("forearm.png", dpi=300)
plt.show()