from simpa.utils import Tags, SegmentationClasses
from simpa.utils.path_manager import PathManager

from simpa.core.simulation import simulate
from simpa.simulation_components import VolumeCreationModelModelBasedAdapter
from simpa.utils import Settings, TISSUE_LIBRARY
from simpa.core.device_digital_twins import RSOMExplorerP50
from simpa.io_handling import load_data_field
import numpy as np
import matplotlib.pyplot as plt
from utils.save_directory import get_save_path
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VOLUME_WIDTH_HEIGHT_DIM_IN_MM = 50
VOLUME_PLANAR_DIM_IN_MM = 50
SPACING = 1.0
RANDOM_SEED = 2736587

path_manager = PathManager()

SAVE_PATH = get_save_path("Tissue_Generation", "Phantom")
VOLUME_NAME = "PhantomScan" + str(RANDOM_SEED)
file_path = SAVE_PATH + "/" + VOLUME_NAME + ".hdf5"


def create_example_tissue(global_settings):
    """
    This is a very simple example script of how to create a tissue definition.
    It contains a muscular background, an epidermis layer on top of the muscles
    and a blood vessel.
    """
    background_dictionary = Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.ultrasound_gel()
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    phantom_material_dictionary = Settings()
    phantom_material_dictionary[Tags.PRIORITY] = 3
    phantom_material_dictionary[Tags.STRUCTURE_START_MM] = [VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2,
                                                            0,
                                                            VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2]
    phantom_material_dictionary[Tags.STRUCTURE_END_MM] = [VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2,
                                                          VOLUME_PLANAR_DIM_IN_MM,
                                                          VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2]
    phantom_material_dictionary[Tags.STRUCTURE_RADIUS_MM] = 14
    phantom_material_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.soft_tissue()
    phantom_material_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = False
    phantom_material_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    inclusion_1_dictionary = Settings()
    inclusion_1_dictionary[Tags.PRIORITY] = 5
    inclusion_1_dictionary[Tags.STRUCTURE_START_MM] = [VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2 + 7 * np.sin(np.deg2rad(10)),
                                                       0,
                                                       VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2 + 7 * np.cos(np.deg2rad(10))]
    inclusion_1_dictionary[Tags.STRUCTURE_END_MM] = [VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2 + 7 * np.sin(np.deg2rad(10)),
                                                     VOLUME_PLANAR_DIM_IN_MM,
                                                     VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2 + 7 * np.cos(np.deg2rad(10))]
    inclusion_1_dictionary[Tags.STRUCTURE_RADIUS_MM] = 2
    inclusion_1_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
    inclusion_1_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = False
    inclusion_1_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    inclusion_2_dictionary = Settings()
    inclusion_2_dictionary[Tags.PRIORITY] = 5
    inclusion_2_dictionary[Tags.STRUCTURE_START_MM] = [VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2 - 7 * np.sin(np.deg2rad(10)),
                                                       0,
                                                       VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2 - 7 * np.cos(np.deg2rad(10))]
    inclusion_2_dictionary[Tags.STRUCTURE_END_MM] = [VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2 - 7 * np.sin(np.deg2rad(10)),
                                                     VOLUME_PLANAR_DIM_IN_MM,
                                                     VOLUME_WIDTH_HEIGHT_DIM_IN_MM / 2 - 7 * np.cos(np.deg2rad(10))]
    inclusion_2_dictionary[Tags.STRUCTURE_RADIUS_MM] = 2
    inclusion_2_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
    inclusion_2_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = False
    inclusion_2_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    tissue_dict = Settings()
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

settings = Settings(settings)

settings.set_volume_creation_settings({
    Tags.STRUCTURES: create_example_tissue(settings),
    Tags.SIMULATE_DEFORMED_LAYERS: False
})


device = RSOMExplorerP50()
SIMUATION_PIPELINE = [
    VolumeCreationModelModelBasedAdapter(settings)
]

simulate(SIMUATION_PIPELINE, settings, device)
wavelength = settings[Tags.WAVELENGTHS][0]

segmentation_mask = load_data_field(file_path=file_path,
                                    wavelength=wavelength,
                                    data_field=Tags.PROPERTY_SEGMENTATION)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.voxels(segmentation_mask==SegmentationClasses.BLOOD, shade=True, facecolors="red", alpha=0.55)
ax.voxels(segmentation_mask==SegmentationClasses.MUSCLE, shade=True, facecolors="yellow", alpha=0.15)
ax.set_aspect('auto')
ax.set_xticks(np.linspace(0, settings[Tags.DIM_VOLUME_X_MM]/settings[Tags.SPACING_MM], 6))
ax.set_yticks(np.linspace(0, settings[Tags.DIM_VOLUME_Y_MM]/settings[Tags.SPACING_MM], 6))
ax.set_zticks(np.linspace(0, settings[Tags.DIM_VOLUME_Z_MM]/settings[Tags.SPACING_MM], 6))
ax.set_xticklabels(np.linspace(0, settings[Tags.DIM_VOLUME_X_MM], 6, dtype=int))
ax.set_yticklabels(np.linspace(0, settings[Tags.DIM_VOLUME_X_MM], 6, dtype=int))
ax.set_zticklabels(np.linspace(0, settings[Tags.DIM_VOLUME_X_MM], 6, dtype=int))
ax.set_zlim(int(settings[Tags.DIM_VOLUME_X_MM]/settings[Tags.SPACING_MM]), 0)
ax.set_zlabel("Depth [mm]")
ax.set_xlabel("x width [mm]")
ax.set_ylabel("y width [mm]")
ax.view_init(elev=10., azim=-45)
plt.savefig(os.path.join(SAVE_PATH, "phantom.svg"), dpi=300)
plt.show()