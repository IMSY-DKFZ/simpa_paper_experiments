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
from utils.basic_settings import create_basic_reconstruction_settings, create_basic_optical_settings, \
    create_basic_acoustic_settings
from simpa.visualisation.matplotlib_data_visualisation import visualise_data

VOLUME_TRANSDUCER_DIM_IN_MM = 90
VOLUME_PLANAR_DIM_IN_MM = 20
VOLUME_HEIGHT_IN_MM = 90
RANDOM_SEED = 500
# spacing_list = [0.3, 0.2, 0.1]
spacing_list = [0.3]
hyperparam_list = ["Default",
                   Tags.MODEL_SENSOR_FREQUENCY_RESPONSE,
                   Tags.ACOUSTIC_SIMULATION_3D,
                   Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION,
                   Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING,
                   Tags.RECONSTRUCTION_MODE_DIFFERENTIAL,
                   ]
# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = PathManager()

WAVELENGTHS = [800]
SAVE_PATH = path_manager.get_hdf5_file_save_path()
SAVE_PATH = os.path.join(SAVE_PATH, "Hyperparameter_Configurations")
os.makedirs(os.path.join(SAVE_PATH), exist_ok=True)

# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.
np.random.seed(RANDOM_SEED)
for spacing in spacing_list:
    for hyperparam in hyperparam_list:
        VOLUME_NAME = "Pipeline_{}_Spacing_{}_Seed_{}".format(hyperparam[0], spacing, RANDOM_SEED)

        general_settings = {
                    # These parameters set the general properties of the simulated volume
                    Tags.RANDOM_SEED: RANDOM_SEED,
                    Tags.VOLUME_NAME: VOLUME_NAME,
                    Tags.SIMULATION_PATH: SAVE_PATH,
                    Tags.SPACING_MM: spacing,
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

        settings.set_optical_settings(create_basic_optical_settings(path_manager))

        settings.set_acoustic_settings(create_basic_acoustic_settings(path_manager))

        settings.set_reconstruction_settings(
            create_basic_reconstruction_settings(path_manager, reconstruction_spacing=settings[Tags.SPACING_MM]))

        pa_device = MSOTAcuityEcho(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                                VOLUME_PLANAR_DIM_IN_MM / 2,
                                                                25]))
        pa_device.update_settings_for_use_of_model_based_volume_creator(settings)

        if hyperparam in [Tags.MODEL_SENSOR_FREQUENCY_RESPONSE,
                          Tags.ACOUSTIC_SIMULATION_3D]:
            settings[Tags.ACOUSTIC_MODEL_SETTINGS][hyperparam] = True
        elif hyperparam in [Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION,
                            Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING]:
            settings[Tags.RECONSTRUCTION_MODEL_SETTINGS][hyperparam] = True
        elif hyperparam == Tags.RECONSTRUCTION_MODE_DIFFERENTIAL:
            settings[Tags.RECONSTRUCTION_MODEL_SETTINGS][Tags.RECONSTRUCTION_MODE] = hyperparam

        SIMUATION_PIPELINE = [
            VolumeCreationModelModelBasedAdapter(settings),
            OpticalForwardModelMcxAdapter(settings),
            AcousticForwardModelKWaveAdapter(settings),
            ImageReconstructionModuleDelayAndSumAdapter(settings),
            FieldOfViewCroppingProcessingComponent(settings),
        ]

        import time
        timer = time.time()
        simulate(SIMUATION_PIPELINE, settings, pa_device)
        VISUALIZE = True
        if VISUALIZE:
            visualise_data(path_to_hdf5_file=SAVE_PATH + "/" + VOLUME_NAME + ".hdf5",
                           wavelength=WAVELENGTHS[0],
                           show_time_series_data=False,
                           show_initial_pressure=True,
                           show_absorption=True,
                           show_segmentation_map=False,
                           show_tissue_density=False,
                           show_reconstructed_data=True,
                           show_fluence=False,
                           log_scale=False,
                           plot_title=hyperparam)
        print("Needed", time.time()-timer, "seconds")
        # TODO global_settings[Tags.SIMPA_OUTPUT_PATH]
        print("Simulating ", RANDOM_SEED, "[Done]")