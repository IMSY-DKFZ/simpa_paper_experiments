import matplotlib.pyplot as plt
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
from simpa.io_handling import load_data_field
from skimage.transform import rescale
from utils.save_directory import get_save_path

VOLUME_TRANSDUCER_DIM_IN_MM = 90
VOLUME_PLANAR_DIM_IN_MM = 20
VOLUME_HEIGHT_IN_MM = 90
RANDOM_SEED = 500
spacing_list = [0.35, 0.25, 0.15]
# spacing_list = [0.3]
hyperparam_list = ["Default",
                   Tags.MODEL_SENSOR_FREQUENCY_RESPONSE,
                   Tags.ACOUSTIC_SIMULATION_3D,
                   Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION,
                   Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING,
                   Tags.RECONSTRUCTION_MODE_DIFFERENTIAL,
                   ]
img_arr = list()
# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = PathManager()

WAVELENGTHS = [800]
SAVE_PATH = get_save_path("pa_image_simulation", "Hyperparameter_Configurations")

# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.
np.random.seed(RANDOM_SEED)
for spacing in spacing_list:
    for i, hyperparam in enumerate(hyperparam_list):
        VOLUME_NAME = "Pipeline_{}_Spacing_{}_Seed_{}".format(hyperparam[0]
                                                              if isinstance(hyperparam, tuple)
                                                              else hyperparam,
                                                              spacing, RANDOM_SEED)

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
            create_basic_reconstruction_settings(path_manager, reconstruction_spacing=min(spacing_list)))

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
        print("Needed", time.time()-timer, "seconds")
        # TODO global_settings[Tags.SIMPA_OUTPUT_PATH]
        print("Simulating ", RANDOM_SEED, "[Done]")

        if i == 0:
            p0 = load_data_field(SAVE_PATH + "/" + VOLUME_NAME + ".hdf5", Tags.OPTICAL_MODEL_INITIAL_PRESSURE, WAVELENGTHS[0])
            p0 = rescale(p0, spacing/min(spacing_list), order=2, anti_aliasing=True)
            norm_p0 = p0 / np.max(p0)
            img_arr.append(norm_p0)

        recon = load_data_field(SAVE_PATH + "/" + VOLUME_NAME + ".hdf5", Tags.RECONSTRUCTED_DATA, WAVELENGTHS[0])
        recon[recon < 0] = 0
        norm_recon = recon / np.max(recon)
        img_arr.append(norm_recon)

from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(figsize=(10, 5))
hyperparam_list.insert(0, "Initial Pressure")
img_grid = ImageGrid(fig, rect=111, nrows_ncols=(len(spacing_list), len(hyperparam_list)), axes_pad=0,
                     cbar_mode="single", cbar_location="right")
for i, gridax in enumerate(img_grid):
    img = np.rot90(img_arr[i], -1)
    # img_shape = img.shape
    # img = img[:int(img_shape[0]/2), :]
    im = gridax.imshow(img)
    if i < len(hyperparam_list):
        gridax.set_title(hyperparam_list[i][0] if isinstance(hyperparam_list[i], tuple) else hyperparam_list[i])
    if i % len(hyperparam_list) == 0:
        gridax.set_ylabel(f"Spacing {spacing_list[int(i / len(hyperparam_list))]}")

img_grid.cbar_axes[0].colorbar(im)
# plt.savefig(SAVE_PATH + "/hyperparam_grid.svg")
plt.show()

