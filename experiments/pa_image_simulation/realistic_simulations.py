"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa.core.simulation import simulate
from simpa.utils.settings import Settings
from simpa.utils import Tags, SegmentationClasses, MolecularCompositionGenerator
import numpy as np
from simpa.utils.libraries.tissue_library import TISSUE_LIBRARY
from simpa.utils.libraries.molecule_library import MOLECULE_LIBRARY
from simpa.visualisation.matplotlib_data_visualisation import visualise_data
from scipy.ndimage import zoom
from simpa.utils.path_manager import PathManager
from simpa.core.device_digital_twins import RSOMExplorerP50, MSOTAcuityEcho
from simpa.core import VolumeCreationModuleSegmentationBasedAdapter, OpticalForwardModelMcxAdapter, \
    GaussianNoiseProcessingComponent, AcousticForwardModelKWaveAdapter, ImageReconstructionModuleDelayAndSumAdapter, \
    FieldOfViewCroppingProcessingComponent
import nrrd
import matplotlib.pyplot as plt

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = True

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = PathManager()
VOLUME_TRANSDUCER_DIM_IN_MM = 80
VOLUME_PLANAR_DIM_IN_MM = 20 #y
VOLUME_HEIGHT_IN_MM = 50 #z
target_spacing = 0.15625

seg, header = nrrd.read("/path/to/segmentation/map")

input_spacing = 0.15625
seg = np.reshape(seg[:, :, 0], (256, 1, 128))
# plt.imshow(seg[:, 0, :])
# plt.show()
segmentation_volume_tiled = np.tile(seg, (1, 128, 1))
segmentation_volume_mask = np.ones((512, 128, 320)) * 6

segmentation_volume_mask[128:384, :, -128:] = segmentation_volume_tiled

segmentation_volume_mask = np.round(zoom(segmentation_volume_mask, input_spacing/target_spacing,
                                         order=0, mode='nearest')).astype(int)
segmentation_volume_mask[segmentation_volume_mask == 0] = 5
# plt.imshow(segmentation_volume_mask[:, 0, :])
# plt.show()

def segmention_class_mapping():
    ret_dict = dict()
    ret_dict[0] = TISSUE_LIBRARY.heavy_water()
    ret_dict[1] = TISSUE_LIBRARY.blood(oxygenation=0.9)
    ret_dict[2] = TISSUE_LIBRARY.epidermis()
    ret_dict[3] = TISSUE_LIBRARY.soft_tissue(blood_volume_fraction=0.05)
    ret_dict[4] = TISSUE_LIBRARY.mediprene()
    ret_dict[5] = TISSUE_LIBRARY.ultrasound_gel()
    ret_dict[6] = TISSUE_LIBRARY.heavy_water()
    # ret_dict[7] = (MolecularCompositionGenerator()
    #                .append(MOLECULE_LIBRARY.oxyhemoglobin(0.01))
    #                .append(MOLECULE_LIBRARY.deoxyhemoglobin(0.01))
    #                .append(MOLECULE_LIBRARY.water(0.98))
    #                .get_molecular_composition(SegmentationClasses.COUPLING_ARTIFACT))
    ret_dict[7] = TISSUE_LIBRARY.muscle(blood_volume_fraction=0.2)
    ret_dict[8] = TISSUE_LIBRARY.blood(oxygenation=0.8)
    ret_dict[9] = TISSUE_LIBRARY.heavy_water()
    ret_dict[10] = TISSUE_LIBRARY.heavy_water()
    return ret_dict


settings = Settings()
settings[Tags.SIMULATION_PATH] = path_manager.get_hdf5_file_save_path()
settings[Tags.VOLUME_NAME] = "SegmentationTest"
settings[Tags.RANDOM_SEED] = 1234
settings[Tags.WAVELENGTHS] = [950]
settings[Tags.SPACING_MM] = target_spacing
settings[Tags.DIM_VOLUME_Z_MM] = VOLUME_HEIGHT_IN_MM
settings[Tags.DIM_VOLUME_X_MM] = VOLUME_TRANSDUCER_DIM_IN_MM
settings[Tags.DIM_VOLUME_Y_MM] = VOLUME_PLANAR_DIM_IN_MM
settings[Tags.GPU] = True

settings.set_volume_creation_settings({
    Tags.INPUT_SEGMENTATION_VOLUME: segmentation_volume_mask,
    Tags.SEGMENTATION_CLASS_MAPPING: segmention_class_mapping(),

})

settings.set_optical_settings({
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e8,
    Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
    Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_MSOT_ACUITY_ECHO,
    Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
})

settings.set_acoustic_settings({
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

settings.set_reconstruction_settings({
    Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: False,
    Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
    # Tags.ACOUSTIC_SIMULATION_3D: True,
    Tags.PROPERTY_ALPHA_POWER: 1.05,
    Tags.TUKEY_WINDOW_ALPHA: 0.5,
    Tags.BANDPASS_CUTOFF_LOWPASS: int(8e6),
    Tags.BANDPASS_CUTOFF_HIGHPASS: int(0.1e6),
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
    Tags.SPACING_MM: 0.15625,
    Tags.MODEL_SENSOR_FREQUENCY_RESPONSE: False,
    Tags.INITIAL_PRESSURE_SMOOTHING: False,
})

settings["noise_initial_pressure"] = {
    Tags.NOISE_MEAN: 1,
    Tags.NOISE_STD: 0.01,
    Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
    Tags.DATA_FIELD: Tags.OPTICAL_MODEL_INITIAL_PRESSURE,
    Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
}

settings["noise_time_series"] = {
    Tags.NOISE_STD: 50,
    Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
    Tags.DATA_FIELD: Tags.TIME_SERIES_DATA
}

pipeline = [
    VolumeCreationModuleSegmentationBasedAdapter(settings),
    OpticalForwardModelMcxAdapter(settings),
    GaussianNoiseProcessingComponent(settings, "noise_initial_pressure"),
    AcousticForwardModelKWaveAdapter(settings),
    GaussianNoiseProcessingComponent(settings, "noise_time_series"),
    ImageReconstructionModuleDelayAndSumAdapter(settings),
    FieldOfViewCroppingProcessingComponent(settings)
]
device = MSOTAcuityEcho(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                     VOLUME_PLANAR_DIM_IN_MM/2,
                                                     -11]),
                        field_of_view_extent_mm=np.asarray([-(2 * np.sin(0.34 / 40 * 128) * 40) / 2,
                                                            (2 * np.sin(0.34 / 40 * 128) * 40) / 2,
                                                            0, 0, 0, 50]))
# device.update_settings_for_use_of_model_based_volume_creator(settings)
simulate(pipeline, settings, device)

if Tags.WAVELENGTH in settings:
    WAVELENGTH = settings[Tags.WAVELENGTH]
else:
    WAVELENGTH = 700

if VISUALIZE:
    visualise_data(path_to_hdf5_file=path_manager.get_hdf5_file_save_path() + "/" + "SegmentationTest" + ".hdf5", wavelength=950,
                   show_absorption=False,
                   show_reconstructed_data=True,
                   log_scale=False,
                   show_time_series_data=False,
                   show_segmentation_map=True,
                   show_initial_pressure=False
                   )
