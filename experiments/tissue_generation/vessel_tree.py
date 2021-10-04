from simpa.utils.libraries.structure_library import VesselStructure, define_vessel_structure_settings
from simpa.utils.libraries.structure_library import HorizontalLayerStructure, define_horizontal_layer_structure_settings
from simpa.utils import TISSUE_LIBRARY
from simpa.utils.deformation_manager import create_deformation_settings
from simpa.utils import Settings, Tags
import matplotlib.pyplot as plt
import numpy as np
from utils.save_directory import get_save_path
import os

SAVE_PATH = get_save_path("Tissue_Generation", "Vessel_Tree")
np.random.seed(1234)

global_settings = Settings()
global_settings[Tags.SPACING_MM] = 0.5
global_settings[Tags.DIM_VOLUME_X_MM] = 50
global_settings[Tags.DIM_VOLUME_Y_MM] = 50
global_settings[Tags.DIM_VOLUME_Z_MM] = 50
global_settings.set_volume_creation_settings({
    Tags.SIMULATE_DEFORMED_LAYERS: True,
    Tags.DEFORMED_LAYERS_SETTINGS: create_deformation_settings(
                bounds_mm=[[0, global_settings[Tags.DIM_VOLUME_X_MM]],
                           [0, global_settings[Tags.DIM_VOLUME_Y_MM]]],
                maximum_z_elevation_mm=10,
                filter_sigma=0,
                cosine_scaling_factor=1)
})
np.random.seed(4315696)
vessel_settings = define_vessel_structure_settings([25, 0, 30], [0, 1, 0],
                                                   molecular_composition=TISSUE_LIBRARY.blood(0.5),
                                                   radius_mm=4, bifurcation_length_mm=15, radius_variation_factor=0,
                                                   curvature_factor=0.1)

vessel = VesselStructure(global_settings, vessel_settings)

vessel_tree = vessel.geometrical_volume

epidermis_settings = define_horizontal_layer_structure_settings(z_start_mm=2, thickness_mm=1,
                                                                adhere_to_deformation=True,
                                                                molecular_composition=TISSUE_LIBRARY.epidermis())

epidermis = HorizontalLayerStructure(global_settings, epidermis_settings)
epidermis_layer = epidermis.geometrical_volume

print("Plotting...")
fontsize = 13
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.voxels(vessel_tree, shade=True, facecolors="red", alpha=0.55)
ax.voxels(epidermis_layer, shade=True, facecolors="brown", alpha=0.55)
ax.set_aspect('auto')
ax.set_xticks(np.linspace(0, global_settings[Tags.DIM_VOLUME_X_MM]/global_settings[Tags.SPACING_MM], 6))
ax.set_yticks(np.linspace(0, global_settings[Tags.DIM_VOLUME_Y_MM]/global_settings[Tags.SPACING_MM], 6))
ax.set_zticks(np.linspace(0, global_settings[Tags.DIM_VOLUME_Z_MM]/global_settings[Tags.SPACING_MM], 6))
ax.set_xticklabels(np.linspace(0, global_settings[Tags.DIM_VOLUME_X_MM], 6, dtype=int), fontsize=fontsize)
ax.set_yticklabels(np.linspace(0, global_settings[Tags.DIM_VOLUME_X_MM], 6, dtype=int), fontsize=fontsize)
ax.set_zticklabels(np.linspace(0, global_settings[Tags.DIM_VOLUME_X_MM], 6, dtype=int), fontsize=fontsize)
ax.set_zlim(int(global_settings[Tags.DIM_VOLUME_X_MM]/global_settings[Tags.SPACING_MM]), 0)
ax.set_zlabel("Depth [mm]", fontsize=fontsize)
ax.set_xlabel("x width [mm]", fontsize=fontsize)
ax.set_ylabel("y width [mm]", fontsize=fontsize)
ax.view_init(elev=10., azim=-45)
plt.savefig(os.path.join(SAVE_PATH, "vessel_tree.svg"), dpi=300)
plt.show()