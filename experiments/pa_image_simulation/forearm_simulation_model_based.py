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
from simpa.visualisation.matplotlib_data_visualisation import visualise_data
import numpy as np
from simpa.utils.path_manager import PathManager
from simpa.core import ImageReconstructionModuleDelayAndSumAdapter, GaussianNoiseProcessingComponent, \
    OpticalForwardModelMcxAdapter, AcousticForwardModelKWaveAdapter, VolumeCreationModelModelBasedAdapter, \
    FieldOfViewCroppingProcessingComponent, ImageReconstructionModuleSignedDelayMultiplyAndSumAdapter, \
    ReconstructionModuleTimeReversalAdapter
from simpa.core.device_digital_twins import MSOTAcuityEcho
from simpa.core.device_digital_twins import LinearArrayDetectionGeometry, SlitIlluminationGeometry, PhotoacousticDevice
import nrrd
from simpa.io_handling import load_data_field
from utils.create_example_tissue import create_realistic_forearm_tissue


# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VOLUME_TRANSDUCER_DIM_IN_MM = 40
VOLUME_PLANAR_DIM_IN_MM = 20
VOLUME_HEIGHT_IN_MM = 20
SPACING = 0.15625
RANDOM_SEED = 1234

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = PathManager()

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = True


orig_im, header = nrrd.read("/path/to/original/file")

number_of_knots = 6

deformation_z_elevations = np.tile([-4, -2, 0, -1.4, -1.8, -2.7][::-1], (5, 1))
deformation_z_elevations = np.moveaxis(deformation_z_elevations, 1, 0)
xx, yy = np.meshgrid(np.linspace(0,  (2 * np.sin(0.34 / 40 * 128) * 40), number_of_knots), np.linspace(0, VOLUME_HEIGHT_IN_MM, 5), indexing='ij')

deformation_settings = dict()
deformation_settings[Tags.DEFORMATION_X_COORDINATES_MM] = xx
deformation_settings[Tags.DEFORMATION_Y_COORDINATES_MM] = yy
deformation_settings[Tags.DEFORMATION_Z_ELEVATIONS_MM] = deformation_z_elevations

# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.

np.random.seed(RANDOM_SEED)
VOLUME_NAME = "CompletePipelineTestMSOT_"+str(RANDOM_SEED)

general_settings = {
            # These parameters set the general properties of the simulated volume
            Tags.RANDOM_SEED: RANDOM_SEED,
            Tags.VOLUME_NAME: "CompletePipelineTestMSOT_" + str(RANDOM_SEED),
            Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
            Tags.SPACING_MM: SPACING,
            Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
            Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
            Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
            Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
            Tags.GPU: True,
            # The following parameters set the optical forward model
            Tags.WAVELENGTHS: [700],
            Tags.LOAD_AND_SAVE_HDF5_FILE_AT_THE_END_OF_SIMULATION_TO_MINIMISE_FILESIZE: True,
            Tags.DO_IPASC_EXPORT: True
        }
settings = Settings(general_settings)
np.random.seed(RANDOM_SEED)

settings.set_volume_creation_settings({
    Tags.STRUCTURES: create_realistic_forearm_tissue(settings),
    Tags.SIMULATE_DEFORMED_LAYERS: True,
    Tags.DEFORMED_LAYERS_SETTINGS: deformation_settings,
    Tags.US_GEL: True
})

settings.set_optical_settings({
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
    Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
    Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO,
    Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
    Tags.MCX_ASSUMED_ANISOTROPY: 0.9,
})

settings.set_acoustic_settings({
    Tags.ACOUSTIC_SIMULATION_3D: False,
    Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
    Tags.PROPERTY_ALPHA_POWER: 0.00,
    Tags.SENSOR_RECORD: "p",
    Tags.PMLInside: False,
    Tags.PMLSize: [31, 32],
    Tags.PMLAlpha: 1.5,
    Tags.PlotPML: False,
    Tags.RECORDMOVIE: False,
    Tags.MOVIENAME: "visualization_log",
    Tags.ACOUSTIC_LOG_SCALE: True
})

settings.set_reconstruction_settings({
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
    Tags.SPACING_MM: SPACING
})

settings["noise_initial_pressure"] = {
    Tags.NOISE_MEAN: 1,
    Tags.NOISE_STD: 0.09,
    Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
    Tags.DATA_FIELD: Tags.OPTICAL_MODEL_INITIAL_PRESSURE,
    Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
}

settings["noise_time_series"] = {
    Tags.NOISE_STD: 10,
    Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
    Tags.DATA_FIELD: Tags.TIME_SERIES_DATA
}

# TODO: For the device choice, uncomment the undesired device

device = MSOTAcuityEcho(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                     VOLUME_PLANAR_DIM_IN_MM/2,
                                                     -3]),
                        field_of_view_extent_mm=np.asarray([-20, 20, 0, 0, 0, 20]))
device.update_settings_for_use_of_model_based_volume_creator(settings)


SIMUATION_PIPELINE = [
    VolumeCreationModelModelBasedAdapter(settings),
    OpticalForwardModelMcxAdapter(settings),
    GaussianNoiseProcessingComponent(settings, "noise_initial_pressure"),
    AcousticForwardModelKWaveAdapter(settings),
    GaussianNoiseProcessingComponent(settings, "noise_time_series"),
    ImageReconstructionModuleDelayAndSumAdapter(settings),
    FieldOfViewCroppingProcessingComponent(settings)
    ]

simulate(SIMUATION_PIPELINE, settings, device)

if Tags.WAVELENGTH in settings:
    WAVELENGTH = settings[Tags.WAVELENGTH]
else:
    WAVELENGTH = 700

recon = load_data_field(path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5",
                        Tags.RECONSTRUCTED_DATA, wavelength=WAVELENGTH)

import matplotlib.pyplot as plt
plt.imshow(np.rot90(recon, 3))
plt.imshow(np.fliplr(np.rot90(orig_im[:, :, 0], 3)), cmap="Greys", alpha=0.5)
plt.show()

