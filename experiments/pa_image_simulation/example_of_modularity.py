from simpa.utils import Tags

from simpa.core.simulation import simulate
from simpa.utils.settings import Settings
import numpy as np
from simpa.utils.path_manager import PathManager
from simpa.core import ImageReconstructionModuleDelayAndSumAdapter, GaussianNoiseProcessingComponent, \
    OpticalForwardModelMcxAdapter, AcousticForwardModelKWaveAdapter, VolumeCreationModelModelBasedAdapter, \
    FieldOfViewCroppingProcessingComponent, ImageReconstructionModuleSignedDelayMultiplyAndSumAdapter, \
    ReconstructionModuleTimeReversalAdapter
from simpa.core.device_digital_twins import MSOTAcuityEcho, InVision256TF
import os
from utils.create_example_tissue import create_square_phantom
import copy

VOLUME_TRANSDUCER_DIM_IN_MM = 75
VOLUME_PLANAR_DIM_IN_MM = 20
VOLUME_HEIGHT_IN_MM = 45
SPACING = 0.25
RANDOM_SEED = 500
loop_end = RANDOM_SEED + 1
# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = PathManager()

# WAVELENGTHS = list(np.arange(700, 855, 10))
WAVELENGTHS = [800]
SAVE_PATH = path_manager.get_hdf5_file_save_path()

devices = {"MSOTAcuityEcho": MSOTAcuityEcho,
           "InVision256TF": InVision256TF}
recon_algorithms = {"DAS": ImageReconstructionModuleDelayAndSumAdapter,
                    "TR": ReconstructionModuleTimeReversalAdapter}
# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.
np.random.seed(RANDOM_SEED)
for device_key, device in devices.items():
    for recon_algorithm_key, recon_algorithm in recon_algorithms.items():

        VOLUME_NAME = "Modularity_example_{}_{}".format(device_key, recon_algorithm_key) + str(RANDOM_SEED)
        os.makedirs(os.path.join(SAVE_PATH, VOLUME_NAME), exist_ok=True)

        general_settings = {
                    # These parameters set the general properties of the simulated volume
                    Tags.RANDOM_SEED: RANDOM_SEED,
                    Tags.VOLUME_NAME: VOLUME_NAME,
                    Tags.SIMULATION_PATH: os.path.join(SAVE_PATH),
                    Tags.SPACING_MM: SPACING,
                    Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
                    Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
                    Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
                    Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
                    Tags.GPU: True,
                    Tags.WAVELENGTHS: WAVELENGTHS,
                    Tags.LOAD_AND_SAVE_HDF5_FILE_AT_THE_END_OF_SIMULATION_TO_MINIMISE_FILESIZE: True
                }
        settings = Settings(general_settings)

        settings.set_volume_creation_settings({
            Tags.STRUCTURES: create_square_phantom(settings),
            Tags.US_GEL: True,

        })

        settings.set_optical_settings({
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 5e7,
            Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
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
            Tags.ACOUSTIC_LOG_SCALE: True,
            Tags.MODEL_SENSOR_FREQUENCY_RESPONSE: False,
            Tags.INITIAL_PRESSURE_SMOOTHING: False,
        })

        settings.set_reconstruction_settings({
            Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: False,
            Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
            # Tags.ACOUSTIC_SIMULATION_3D: True,
            Tags.PROPERTY_ALPHA_POWER: 0.0,
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
            Tags.SPACING_MM: 0.1,
            Tags.MODEL_SENSOR_FREQUENCY_RESPONSE: False
        })

        settings["noise_initial_pressure"] = {
            Tags.NOISE_MEAN: 1,
            Tags.NOISE_STD: 0.09,
            Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
            Tags.DATA_FIELD: Tags.OPTICAL_MODEL_INITIAL_PRESSURE,
            Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
        }

        settings["noise_time_series"] = {
            Tags.NOISE_STD: 50,
            Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
            Tags.DATA_FIELD: Tags.TIME_SERIES_DATA
        }

        pa_device = device(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                        VOLUME_PLANAR_DIM_IN_MM/2,
                                                        0]))
        if isinstance(pa_device, MSOTAcuityEcho):
            pa_device.update_settings_for_use_of_model_based_volume_creator(settings)

        SIMUATION_PIPELINE = [
            VolumeCreationModelModelBasedAdapter(settings),
            OpticalForwardModelMcxAdapter(settings),
            GaussianNoiseProcessingComponent(settings, "noise_initial_pressure"),
            AcousticForwardModelKWaveAdapter(settings),
            GaussianNoiseProcessingComponent(settings, "noise_time_series"),
            recon_algorithm(settings),
            FieldOfViewCroppingProcessingComponent(settings)
            ]

        import time
        timer = time.time()
        simulate(SIMUATION_PIPELINE, settings, pa_device)

        print("Needed", time.time()-timer, "seconds")
        # TODO global_settings[Tags.SIMPA_OUTPUT_PATH]
        print("Simulating ", RANDOM_SEED, "[Done]")
