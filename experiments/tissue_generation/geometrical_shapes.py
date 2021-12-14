import os
from simpa import Tags
import simpa as sp
import matplotlib.pyplot as plt
import numpy as np
from utils.save_directory import get_save_path

SAVE_PATH = get_save_path("Tissue_Generation", "Geometrical_Shapes")

np.random.seed(28374628)

global_settings = sp.Settings()
global_settings[Tags.SPACING_MM] = 0.5
global_settings[Tags.DIM_VOLUME_X_MM] = 50
global_settings[Tags.DIM_VOLUME_Y_MM] = 50
global_settings[Tags.DIM_VOLUME_Z_MM] = 50
global_settings.set_volume_creation_settings({
    Tags.SIMULATE_DEFORMED_LAYERS: True,
    Tags.DEFORMED_LAYERS_SETTINGS: sp.create_deformation_settings(
                bounds_mm=[[0, global_settings[Tags.DIM_VOLUME_X_MM]],
                           [0, global_settings[Tags.DIM_VOLUME_Y_MM]]],
                maximum_z_elevation_mm=10,
                filter_sigma=0,
                cosine_scaling_factor=1)
})
import time
start_time = time.time()
sphere_settings = sp.define_spherical_structure_settings(start_mm=[25, 20, 33], radius_mm=10,
                                                         molecular_composition=sp.TISSUE_LIBRARY.epidermis())
sphere = sp.SphericalStructure(global_settings, sphere_settings).geometrical_volume

tube_settings = sp.define_circular_tubular_structure_settings(tube_start_mm=[10, 0, 30], tube_end_mm=[15, 50, 10],
                                                              radius_mm=5,
                                                              molecular_composition=sp.TISSUE_LIBRARY.epidermis())
tube = sp.CircularTubularStructure(global_settings, tube_settings).geometrical_volume

parallel_settings = sp.define_parallelepiped_structure_settings(start_mm=[8, 40, 25],
                                                                edge_a_mm=[25, 0, 5],
                                                                edge_b_mm=[10, 5, -5],
                                                                edge_c_mm=[10, 0, 15],
                                                                molecular_composition=sp.TISSUE_LIBRARY.epidermis())
parallelepiped = sp.ParallelepipedStructure(global_settings, parallel_settings).geometrical_volume

cube_settings = sp.define_rectangular_cuboid_structure_settings(start_mm=[50, 0, 50],
                                                                extent_mm=[-10, 10, -10],
                                                                molecular_composition=sp.TISSUE_LIBRARY.epidermis())
cube = sp.RectangularCuboidStructure(global_settings, cube_settings).geometrical_volume
end_time = time.time() - start_time
with open(os.path.join(SAVE_PATH, "run_time.txt"), "w+") as out_file:
    out_file.write("{:.2f} s".format(end_time))
print("Plotting...")
fontsize = 13
plt.rcParams["axes.grid"] = True
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.voxels(sphere, shade=True, facecolors="blue", alpha=0.55)
ax.voxels(tube, shade=True, facecolors="green", alpha=0.55)
ax.voxels(parallelepiped, shade=True, facecolors="yellow", alpha=0.55)
ax.voxels(cube, shade=True, facecolors="red", alpha=0.55)
ax.set_aspect('auto')
# ax.set_xticks(np.linspace(0, global_settings[Tags.DIM_VOLUME_X_MM]/global_settings[Tags.SPACING_MM], 6))
# ax.set_yticks(np.linspace(0, global_settings[Tags.DIM_VOLUME_Y_MM]/global_settings[Tags.SPACING_MM], 6))
# ax.set_zticks(np.linspace(0, global_settings[Tags.DIM_VOLUME_Z_MM]/global_settings[Tags.SPACING_MM], 6))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# ax.set_xticklabels(np.linspace(0, global_settings[Tags.DIM_VOLUME_X_MM], 6, dtype=int), fontsize=fontsize)
# ax.set_yticklabels(np.linspace(0, global_settings[Tags.DIM_VOLUME_X_MM], 6, dtype=int), fontsize=fontsize)
# ax.set_zticklabels(np.linspace(0, global_settings[Tags.DIM_VOLUME_X_MM], 6, dtype=int), fontsize=fontsize)
ax.set_zlim(int(global_settings[Tags.DIM_VOLUME_X_MM]/global_settings[Tags.SPACING_MM]), 0)
# ax.set_zlabel("Depth [mm]", fontsize=fontsize)
# ax.set_xlabel("x width [mm]", fontsize=fontsize)
# ax.set_ylabel("y width [mm]", fontsize=fontsize)
ax.view_init(elev=10., azim=-45)
plt.savefig(os.path.join(SAVE_PATH, "geometrical_shapes.svg"), dpi=300)
plt.close()
