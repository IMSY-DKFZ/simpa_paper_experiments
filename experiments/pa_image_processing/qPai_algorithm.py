"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from simpa.utils import Tags, TISSUE_LIBRARY
from simpa.core.simulation import simulate
from simpa.io_handling import load_hdf5, load_data_field
from simpa.utils.settings import Settings
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from simpa.core import VolumeCreationModelModelBasedAdapter, OpticalForwardModelMcxAdapter, \
    GaussianNoiseProcessingComponent
from simpa.algorithms.monospectral.iterative_qPAI_algorithm import IterativeqPAIProcessingComponent
from simpa.utils.path_manager import PathManager
from simpa.core.device_digital_twins import PhotoacousticDevice, PencilBeamIlluminationGeometry, GaussianBeamIlluminationGeometry
from utils.create_example_tissue import create_qPAI_phantom


# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = PathManager()
SAVE_PATH = path_manager.get_hdf5_file_save_path()
SAVE_PATH = os.path.join(SAVE_PATH, "qPAI_Algorithm")
os.makedirs(os.path.join(SAVE_PATH), exist_ok=True)

VOLUME_TRANSDUCER_DIM_IN_MM = 12
VOLUME_PLANAR_DIM_IN_MM = 10
VOLUME_HEIGHT_IN_MM = 12
SPACING = 0.08
RANDOM_SEED = 500
VOLUME_NAME = "qPAI_Algorithm" + str(RANDOM_SEED)

# If VISUALIZE is set to True, the reconstruction result will be plotted
VISUALIZE = True

# set settings for volume creation, optical simulation and iterative qPAI method
np.random.seed(RANDOM_SEED)

general_settings = {
    # These parameters set the general properties of the simulated volume
    Tags.RANDOM_SEED: RANDOM_SEED,
    Tags.VOLUME_NAME: VOLUME_NAME,
    Tags.SIMULATION_PATH: SAVE_PATH,
    Tags.SPACING_MM: SPACING,
    Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
    Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
    Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
    Tags.WAVELENGTHS: [700]
}

settings = Settings(general_settings)

settings.set_volume_creation_settings({
    # These parameters set the properties for the volume creation
    Tags.STRUCTURES: create_qPAI_phantom(settings)
})
settings.set_optical_settings({
    # These parameters set the properties for the optical Monte Carlo simulation
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
    Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
    Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
    Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50
})
settings["noise_model"] = {
    Tags.NOISE_MEAN: 1.0,
    Tags.NOISE_STD: 0.01,
    Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
    Tags.DATA_FIELD: Tags.OPTICAL_MODEL_INITIAL_PRESSURE,
    Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
}
settings["iterative_qpai_reconstruction"] = {
    # These parameters set the properties of the iterative reconstruction
    Tags.DOWNSCALE_FACTOR: 0.473,
    Tags.ITERATIVE_RECONSTRUCTION_CONSTANT_REGULARIZATION: False,
    # the following tag has no effect, since the regularization is chosen to be SNR dependent, not constant
    Tags.ITERATIVE_RECONSTRUCTION_REGULARIZATION_SIGMA: 0.01,
    Tags.ITERATIVE_RECONSTRUCTION_MAX_ITERATION_NUMBER: 20,
    # for this example, we are not interested in all absorption updates
    Tags.ITERATIVE_RECONSTRUCTION_SAVE_INTERMEDIATE_RESULTS: False,
    Tags.ITERATIVE_RECONSTRUCTION_STOPPING_LEVEL: 0.03
}

# run pipeline including iterative qPAI method
pipeline = [
    VolumeCreationModelModelBasedAdapter(settings),
    OpticalForwardModelMcxAdapter(settings),
    GaussianNoiseProcessingComponent(settings, "noise_model"),
    IterativeqPAIProcessingComponent(settings, "iterative_qpai_reconstruction")
]


class CustomDevice(PhotoacousticDevice):

    def __init__(self):
        super(CustomDevice, self).__init__(device_position_mm=np.asarray([general_settings[Tags.DIM_VOLUME_X_MM] / 2,
                                                                          general_settings[Tags.DIM_VOLUME_Y_MM] / 2,
                                                                          0]))
        self.add_illumination_geometry(GaussianBeamIlluminationGeometry())


device = CustomDevice()

device.update_settings_for_use_of_model_based_volume_creator(settings)

simulate(pipeline, settings, device)

# visualize reconstruction results
if VISUALIZE:
    # get simulation output
    data_path = SAVE_PATH + "/" + VOLUME_NAME + ".hdf5"
    file = load_hdf5(data_path)
    settings = Settings(file["settings"])
    wavelength = settings[Tags.WAVELENGTHS][0]

    # get reconstruction result
    absorption_reconstruction = load_data_field(data_path, Tags.ITERATIVE_qPAI_RESULT, wavelength)

    # get ground truth absorption coefficients
    absorption_gt = load_data_field(data_path, Tags.PROPERTY_ABSORPTION_PER_CM, wavelength)
    fluence = load_data_field(data_path, Tags.OPTICAL_MODEL_FLUENCE, wavelength)

    # rescale ground truth to same dimension as reconstruction (necessary due to resampling in iterative algorithm)
    scale = np.shape(absorption_reconstruction)[0] / np.shape(absorption_gt)[0]  # same as Tags.DOWNSCALE_FACTOR
    absorption_gt = zoom(absorption_gt, scale, order=0, mode="nearest")

    # compute reconstruction error
    difference = absorption_gt - absorption_reconstruction

    median_error = np.median(difference)
    q3, q1 = np.percentile(difference, [75, 25])
    iqr = q3 - q1

    # visualize results
    x_pos = int(np.shape(absorption_gt)[0] / 2)
    y_pos = int(np.shape(absorption_gt)[1] / 2)

    if np.min(absorption_gt) > np.min(absorption_reconstruction):
        cmin = np.min(absorption_reconstruction)
    else:
        cmin = np.min(absorption_gt)

    if np.max(absorption_gt) > np.max(absorption_reconstruction):
        cmax = np.max(absorption_gt)
    else:
        cmax = np.max(absorption_reconstruction)

    results_x_z = [absorption_gt[:, y_pos, :], absorption_reconstruction[:, y_pos, :], difference[:, y_pos, :]]
    results_y_z = [absorption_gt[x_pos, :, :], absorption_reconstruction[x_pos, :, :], difference[x_pos, :, :]]

    label = ["Absorption coefficients: ${\mu_a}^{gt}$", "Reconstruction: ${\mu_a}^{reconstr.}$",
             "Difference: ${\mu_a}^{gt} - {\mu_a}^{reconstr.}$"]

    plt.figure(figsize=(20, 15))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Iterative qPAI Reconstruction \n median error = " + str(np.round(median_error, 4)) +
                 "\n IQR = " + str(np.round(iqr, 4)), fontsize=10)

    cmap = "viridis"

    for i, quantity in enumerate(results_x_z):
        plt.subplot(1, len(results_x_z) + 1, i + 1)
        if i == 0:
            plt.ylabel("x-z", fontsize=10)
        plt.title(label[i], fontsize=10)
        if i == 2:
            cmap = "Reds"
        plt.imshow(np.rot90(quantity, -1), cmap=cmap)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.colorbar()
        if i != 2:
            plt.clim(cmin, cmax)
        else:
            plt.clim(np.min(difference), np.max(difference))
    plt.subplot(1, len(results_x_z) + 1, i+1)
    plt.imshow(fluence[:, y_pos, :])
    plt.show()
    plt.close()
