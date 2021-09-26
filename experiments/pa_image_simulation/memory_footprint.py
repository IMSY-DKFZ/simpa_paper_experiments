from simpa.utils import Tags
from simpa.log import Logger
from simpa.core.simulation import simulate
from simpa.utils.settings import Settings
import numpy as np
from simpa.utils.path_manager import PathManager
from simpa.simulation_components import ImageReconstructionModuleDelayAndSumAdapter, GaussianNoiseProcessingComponent, \
    OpticalForwardModelMcxAdapter, AcousticForwardModelKWaveAdapter, VolumeCreationModelModelBasedAdapter, \
    FieldOfViewCroppingProcessingComponent, ImageReconstructionModuleSignedDelayMultiplyAndSumAdapter, \
    ReconstructionModuleTimeReversalAdapter
from simpa.core.device_digital_twins import MSOTAcuityEcho, InVision256TF
import os
from utils.create_example_tissue import create_square_phantom
from utils.basic_settings import create_basic_reconstruction_settings, create_basic_optical_settings, \
    create_basic_acoustic_settings
from argparse import ArgumentParser
import time
from utils.save_directory import get_save_path

parser = ArgumentParser()
parser.add_argument("--spacing", type=float, default=0.4, help="Spacing for the simulation")
parser.add_argument("--run", type=int, default=0,
                    help="Number that specifies how many times this spacing has been repeated")
config = parser.parse_args()

VOLUME_TRANSDUCER_DIM_IN_MM = 90
VOLUME_PLANAR_DIM_IN_MM = 20
VOLUME_HEIGHT_IN_MM = 90
RANDOM_SEED = 500
# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = PathManager()

WAVELENGTHS = [800]
SAVE_PATH = get_save_path("pa_image_simulation", "Memory_Footprint")
spacing = config.spacing
run = config.run
logger = Logger(os.path.join(SAVE_PATH, "simpa_spacing_{}_run_{}.log".format(spacing, run)), force_new_instance=True)
# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.
np.random.seed(RANDOM_SEED)

VOLUME_NAME = "Spacing_{}_Run_{}_Seed_{}".format(spacing, run, RANDOM_SEED)

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
                                                        0]),
                           field_of_view_extent_mm=np.asarray([-(2 * np.sin(0.34 / 40 * 128) * 40) / 2,
                                                               (2 * np.sin(0.34 / 40 * 128) * 40) / 2,
                                                               0, 0, 0, 100]))
pa_device.update_settings_for_use_of_model_based_volume_creator(settings)

SIMUATION_PIPELINE = [
    VolumeCreationModelModelBasedAdapter(settings),
    OpticalForwardModelMcxAdapter(settings),
    AcousticForwardModelKWaveAdapter(settings),
    ImageReconstructionModuleDelayAndSumAdapter(settings),
    FieldOfViewCroppingProcessingComponent(settings)
    ]

timer = time.time()
simulate(SIMUATION_PIPELINE, settings, pa_device)

print("Needed", time.time()-timer, "seconds")
# TODO global_settings[Tags.SIMPA_OUTPUT_PATH]
print("Simulating ", RANDOM_SEED, "[Done]")

