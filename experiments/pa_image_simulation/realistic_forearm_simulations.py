"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.utils import Tags, TISSUE_LIBRARY
from simpa.utils.libraries.structure_library import define_horizontal_layer_structure_settings,\
    define_circular_tubular_structure_settings, define_elliptical_tubular_structure_settings
from simpa.core.simulation import simulate
from simpa.utils.settings import Settings
import matplotlib.pyplot as plt
import numpy as np
from simpa.utils.path_manager import PathManager
from simpa.simulation_components import ImageReconstructionModuleDelayAndSumAdapter, GaussianNoiseProcessingComponent, \
    OpticalForwardModelMcxAdapter, AcousticForwardModelKWaveAdapter, VolumeCreationModelModelBasedAdapter, \
    FieldOfViewCroppingProcessingComponent, VolumeCreationModuleSegmentationBasedAdapter, \
    ReconstructionModuleTimeReversalAdapter
from simpa.core.device_digital_twins import MSOTAcuityEcho
from copy import deepcopy
import nrrd
from simpa.io_handling import load_data_field
from utils.create_example_tissue import create_realistic_forearm_tissue, create_forearm_segmentation_tissue, \
    segmention_class_mapping
from matplotlib_scalebar.scalebar import ScaleBar
from visualization.colorbar import col_bar
from utils.normalizations import normalize_min_max
from utils.save_directory import get_save_path

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
path_manager = PathManager()
SAVE_PATH = get_save_path("pa_image_simulation", "Realistic_Forearm_Simulation")

REAL_IMAGE_PATH = "/path/to/real/image"
SEGMENTATION_MASK_PATH = "/path/to/segmentation/mask"
SPACING = 0.15625
RANDOM_SEED = 1234
WAVELENGTHS = [700]
SHOW_IMAGE = False

simulation_dictionary = {"Segmentation-based image": dict(),
                         "Model-based image": dict()}

seg_settings = Settings()
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

seg_settings.set_optical_settings({
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
    Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
    Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
    Tags.MCX_ASSUMED_ANISOTROPY: 0.9,
})

seg_settings.set_acoustic_settings({
    Tags.ACOUSTIC_SIMULATION_3D: False,
    Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
    Tags.PROPERTY_ALPHA_POWER: 1.05,
    Tags.SENSOR_RECORD: "p",
    Tags.PMLInside: False,
    Tags.PMLSize: [31, 32],
    Tags.PMLAlpha: 1.5,
    Tags.PlotPML: False,
    Tags.RECORDMOVIE: False,
    Tags.MOVIENAME: "visualization_log",
    Tags.ACOUSTIC_LOG_SCALE: True,
    Tags.MODEL_SENSOR_FREQUENCY_RESPONSE: False,
    Tags.INITIAL_PRESSURE_SMOOTHING: True,
})

seg_settings.set_reconstruction_settings({
    Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: False,
    Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
    Tags.ACOUSTIC_SIMULATION_3D: False,
    Tags.PROPERTY_ALPHA_POWER: 0.00,
    Tags.TUKEY_WINDOW_ALPHA: 0.5,
    Tags.BANDPASS_CUTOFF_LOWPASS: int(8e6),
    Tags.BANDPASS_CUTOFF_HIGHPASS: int(0.1e4),
    Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: True,
    Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
    Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
    Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_DIFFERENTIAL,
    Tags.SENSOR_RECORD: "p",
    Tags.PMLInside: False,
    Tags.PMLSize: [31, 32],
    Tags.PMLAlpha: 1.5,
    Tags.PlotPML: False,
    Tags.RECORDMOVIE: False,
    Tags.MOVIENAME: "visualization_log",
    Tags.ACOUSTIC_LOG_SCALE: True,
    Tags.PROPERTY_SPEED_OF_SOUND: 1540,
    Tags.PROPERTY_ALPHA_COEFF: 0.01,
    Tags.PROPERTY_DENSITY: 1000,
    Tags.SPACING_MM: SPACING,
    Tags.MODEL_SENSOR_FREQUENCY_RESPONSE: False,
    Tags.INITIAL_PRESSURE_SMOOTHING: False,
})

seg_settings["noise_initial_pressure"] = {
    Tags.NOISE_MEAN: 1,
    Tags.NOISE_STD: 0.05,
    Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
    Tags.DATA_FIELD: Tags.OPTICAL_MODEL_INITIAL_PRESSURE,
    Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
}

seg_settings["noise_time_series"] = {
    Tags.NOISE_STD: 25,
    Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
    Tags.DATA_FIELD: Tags.TIME_SERIES_DATA
}

segmentation_volume_mask = create_forearm_segmentation_tissue(
    SEGMENTATION_MASK_PATH,
    spacing=SPACING)

seg_settings.set_volume_creation_settings({
    Tags.INPUT_SEGMENTATION_VOLUME: segmentation_volume_mask,
    Tags.SEGMENTATION_CLASS_MAPPING: segmention_class_mapping()
})

simulation_dictionary["Segmentation-based image"]["settings"] = seg_settings
simulation_dictionary["Segmentation-based image"]["device"] = MSOTAcuityEcho(
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
simulation_dictionary["Model-based image"]["device"] = MSOTAcuityEcho(
    device_position_mm=np.array([40/2, 20/2, -3]),
    field_of_view_extent_mm=np.asarray([-20, 20, 0, 0, 0, 20]))

for i, (mode, dictonary) in enumerate(simulation_dictionary.items()):
    settings = dictonary["settings"]
    device = dictonary["device"]
    if mode == "Model-based image":
        device.update_settings_for_use_of_model_based_volume_creator(settings)
        volume_creation_adapter = VolumeCreationModelModelBasedAdapter
    else:
        settings["noise_time_series"][Tags.NOISE_STD] = 70
        volume_creation_adapter = VolumeCreationModuleSegmentationBasedAdapter

    SIMUATION_PIPELINE = [
        volume_creation_adapter(settings),
        OpticalForwardModelMcxAdapter(settings),
        GaussianNoiseProcessingComponent(settings, "noise_initial_pressure"),
        AcousticForwardModelKWaveAdapter(settings),
        GaussianNoiseProcessingComponent(settings, "noise_time_series"),
        ImageReconstructionModuleDelayAndSumAdapter(settings),
        FieldOfViewCroppingProcessingComponent(settings)
    ]

    # simulate(SIMUATION_PIPELINE, settings, device)

    recon = load_data_field(SAVE_PATH + "/" + settings[Tags.VOLUME_NAME] + ".hdf5",
                            Tags.RECONSTRUCTED_DATA, wavelength=WAVELENGTHS[0])

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

orig_im, header = nrrd.read(REAL_IMAGE_PATH)
orig_im = normalize_min_max(orig_im[:, :, 0])
img_to_plot = np.fliplr(np.rot90(orig_im, 3))[:-1, :]
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
