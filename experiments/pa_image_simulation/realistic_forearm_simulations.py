# SPDX-FileCopyrightText: 2021 Intelligent Medical Systems Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa import Tags
import simpa as sp
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import nrrd
from utils.create_example_tissue import create_realistic_forearm_tissue, create_forearm_segmentation_tissue, \
    segmention_class_mapping
from matplotlib_scalebar.scalebar import ScaleBar
from utils.normalizations import normalize_min_max
from utils.save_directory import get_save_path
from utils.basic_settings import create_basic_optical_settings, create_basic_acoustic_settings, \
    create_basic_reconstruction_settings
from zipfile import ZipFile
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
path_manager = sp.PathManager()
SAVE_PATH = get_save_path("pa_image_simulation", "Realistic_Forearm_Simulation")
example_data_folder = "example_data"
example_data_folder_zip = f"{example_data_folder}.zip"

if example_data_folder_zip not in os.listdir(SAVE_PATH):

    sp.download_from_zenodo(record_id="941302", save_dir=SAVE_PATH, sandbox=True)

if example_data_folder not in os.listdir(SAVE_PATH):
    with ZipFile(os.path.join(SAVE_PATH, f"{example_data_folder}.zip")) as zip_ref:
        zip_ref.extractall(os.path.join(SAVE_PATH))

REAL_IMAGE_PATH = os.path.join(SAVE_PATH, example_data_folder, "real_pa_forearm_image.nrrd")
SEGMENTATION_MASK_PATH = os.path.join(SAVE_PATH, example_data_folder, "real_pa_forearm_image_annotated.nrrd")
SPACING = 0.15625
RANDOM_SEED = 1234
WAVELENGTHS = [700]
SHOW_IMAGE = False

simulation_dictionary = {"Segmentation-based image": dict(),
                         "Model-based image": dict()}

seg_settings = sp.Settings()
seg_settings[Tags.SIMULATION_PATH] = SAVE_PATH
seg_settings[Tags.VOLUME_NAME] = "Segmentation-based image"
seg_settings[Tags.RANDOM_SEED] = RANDOM_SEED
seg_settings[Tags.WAVELENGTHS] = WAVELENGTHS
seg_settings[Tags.SPACING_MM] = SPACING
seg_settings[Tags.DIM_VOLUME_Z_MM] = 50
seg_settings[Tags.DIM_VOLUME_X_MM] = 80
seg_settings[Tags.DIM_VOLUME_Y_MM] = 20
seg_settings[Tags.GPU] = True

np.random.seed(RANDOM_SEED)

seg_settings.set_optical_settings(create_basic_optical_settings(path_manager))

seg_settings.set_acoustic_settings(create_basic_acoustic_settings(path_manager))

seg_settings.set_reconstruction_settings(create_basic_reconstruction_settings(path_manager, SPACING))
seg_settings[Tags.RECONSTRUCTION_MODEL_SETTINGS][Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION] = True
seg_settings[Tags.RECONSTRUCTION_MODEL_SETTINGS][Tags.RECONSTRUCTION_MODE] = Tags.RECONSTRUCTION_MODE_DIFFERENTIAL
seg_settings[Tags.RECONSTRUCTION_MODEL_SETTINGS][Tags.SPACING_MM] = 0.1

seg_settings["noise_initial_pressure"] = {
    Tags.NOISE_MEAN: 1,
    Tags.NOISE_STD: 0.05,
    Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
    Tags.DATA_FIELD: Tags.DATA_FIELD_INITIAL_PRESSURE,
    Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
}

seg_settings["noise_time_series"] = {
    Tags.NOISE_STD: 25,
    Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
    Tags.DATA_FIELD: Tags.DATA_FIELD_TIME_SERIES_DATA
}

segmentation_volume_mask = create_forearm_segmentation_tissue(
    SEGMENTATION_MASK_PATH,
    spacing=SPACING)

seg_settings.set_volume_creation_settings({
    Tags.INPUT_SEGMENTATION_VOLUME: segmentation_volume_mask,
    Tags.SEGMENTATION_CLASS_MAPPING: segmention_class_mapping()
})

simulation_dictionary["Segmentation-based image"]["settings"] = seg_settings
simulation_dictionary["Segmentation-based image"]["device"] = sp.MSOTAcuityEcho(
    device_position_mm=np.array([80/2, 20/2, -11]),
    field_of_view_extent_mm=np.asarray([-20, 20, 0, 0, -2.3, 17.7]))

model_settings = deepcopy(seg_settings)
model_settings[Tags.VOLUME_NAME] = "Model-based image"
model_settings[Tags.DIM_VOLUME_Z_MM] = 20
model_settings[Tags.DIM_VOLUME_X_MM] = 40
model_settings[Tags.DIM_VOLUME_Y_MM] = 20
model_settings[Tags.GPU] = True

number_of_knots = 6

deformation_z_elevations = np.tile([-4, -2, 0, -1.4, -1.8, -2.7][::-1], (5, 1))
deformation_z_elevations = np.moveaxis(deformation_z_elevations, 1, 0)
xx, yy = np.meshgrid(np.linspace(0,  (2 * np.sin(0.34 / 40 * 128) * 40), number_of_knots), np.linspace(0, 20, 5),
                     indexing='ij')

deformation_settings = dict()
deformation_settings[Tags.DEFORMATION_X_COORDINATES_MM] = xx
deformation_settings[Tags.DEFORMATION_Y_COORDINATES_MM] = yy
deformation_settings[Tags.DEFORMATION_Z_ELEVATIONS_MM] = deformation_z_elevations

model_settings.set_volume_creation_settings({
    Tags.STRUCTURES: create_realistic_forearm_tissue(model_settings),
    Tags.SIMULATE_DEFORMED_LAYERS: True,
    Tags.DEFORMED_LAYERS_SETTINGS: deformation_settings,
    Tags.US_GEL: True
})

simulation_dictionary["Model-based image"]["settings"] = model_settings
simulation_dictionary["Model-based image"]["device"] = sp.MSOTAcuityEcho(
    device_position_mm=np.array([40/2, 20/2, -3]),
    field_of_view_extent_mm=np.asarray([-20, 20, 0, 0, 0, 20]))

start_time = time.time()
for i, (mode, dictonary) in enumerate(simulation_dictionary.items()):
    settings = dictonary["settings"]
    device = dictonary["device"]
    if mode == "Model-based image":
        device.update_settings_for_use_of_model_based_volume_creator(settings)
        volume_creation_adapter = sp.ModelBasedVolumeCreationAdapter
    else:
        settings["noise_time_series"][Tags.NOISE_STD] = 70
        volume_creation_adapter = sp.SegmentationBasedVolumeCreationAdapter

    SIMUATION_PIPELINE = [
        volume_creation_adapter(settings),
        sp.MCXAdapter(settings),
        sp.GaussianNoise(settings, "noise_initial_pressure"),
        sp.KWaveAdapter(settings),
        sp.GaussianNoise(settings, "noise_time_series"),
        sp.DelayAndSumAdapter(settings),
        sp.FieldOfViewCropping(settings)
    ]

    sp.simulate(SIMUATION_PIPELINE, settings, device)

    recon = sp.load_data_field(SAVE_PATH + "/" + settings[Tags.VOLUME_NAME] + ".hdf5",
                            Tags.DATA_FIELD_RECONSTRUCTED_DATA, wavelength=WAVELENGTHS[0])

    recon = normalize_min_max(recon)
    img_to_plot = np.rot90(recon, 3)
    print(img_to_plot.shape)
    img_plot = plt.imshow(img_to_plot)
    scale_bar = ScaleBar(SPACING, units="mm", location="lower left", font_properties={"family": "Cmr10", "size": 20})
    plt.gca().add_artist(scale_bar)
    plt.axis("off")
    if SHOW_IMAGE:
        plt.show()
        plt.close()
    else:
        plt.savefig(SAVE_PATH + "/{}.svg".format(mode))
        plt.close()
end_time = time.time() - start_time
with open(os.path.join(SAVE_PATH, "run_time.txt"), "w+") as out_file:
    out_file.write("{:.2f} s".format(end_time))

orig_im, header = nrrd.read(REAL_IMAGE_PATH)
orig_im = normalize_min_max(orig_im[:, :, 0])
img_to_plot = np.fliplr(np.rot90(orig_im, 3))
img_plot = plt.imshow(img_to_plot)
print(img_to_plot.shape)
scale_bar = ScaleBar(SPACING, units="mm", location="lower left", font_properties={"family": "Cmr10", "size": 20})
plt.gca().add_artist(scale_bar)
plt.axis("off")

if SHOW_IMAGE:
    plt.show()
    plt.close()
else:
    plt.savefig(SAVE_PATH + "/real_im.svg")
    plt.close()
    fontsize = 30
    fontname = "Cmr10"
    fig = plt.figure()
    ax = fig.add_axes([0.01, 0.02, 0.05, 0.95])
    import matplotlib as mpl
    cbar = mpl.colorbar.ColorbarBase(ax)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family(fontname)
    cbar.ax.tick_params(labelsize=fontsize)
    plt.savefig(SAVE_PATH + "/cbar_realistic.svg", bbox_inches="tight")
