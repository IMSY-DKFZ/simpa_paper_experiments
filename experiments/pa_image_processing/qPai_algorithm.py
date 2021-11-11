"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

from simpa import Tags
import simpa as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from utils.create_example_tissue import create_qPAI_phantom
from utils.save_directory import get_save_path
from visualization.colorbar import col_bar
from matplotlib_scalebar.scalebar import ScaleBar
from utils.normalizations import normalize_min_max
# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = sp.PathManager()
SAVE_PATH = get_save_path("pa_image_processing", "qPai_algorithm")

VOLUME_TRANSDUCER_DIM_IN_MM = 12
VOLUME_PLANAR_DIM_IN_MM = 10
VOLUME_HEIGHT_IN_MM = 12
SPACING = 0.08
RANDOM_SEED = 500
VOLUME_NAME = "qPAI_Algorithm" + str(RANDOM_SEED)

# If SHOW_IMAGE is set to True, the reconstruction result will be shown instead of saved
SHOW_IMAGE = False

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

settings = sp.Settings(general_settings)

settings.set_volume_creation_settings({
    # These parameters set the properties for the volume creation
    Tags.STRUCTURES: create_qPAI_phantom(settings)
})
settings.set_optical_settings({
    # These parameters set the properties for the optical Monte Carlo simulation
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
    Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
    Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
    Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
    Tags.MCX_ASSUMED_ANISOTROPY: 0.8
})
settings["noise_model"] = {
    Tags.NOISE_MEAN: 1.0,
    Tags.NOISE_STD: 0.01,
    Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
    Tags.DATA_FIELD: Tags.DATA_FIELD_INITIAL_PRESSURE,
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
    sp.ModelBasedVolumeCreationAdapter(settings),
    sp.MCXAdapter(settings),
    sp.GaussianNoise(settings, "noise_model"),
    sp.IterativeqPAI(settings, "iterative_qpai_reconstruction")
]


class CustomDevice(sp.PhotoacousticDevice):

    def __init__(self):
        super(CustomDevice, self).__init__(device_position_mm=np.asarray([general_settings[Tags.DIM_VOLUME_X_MM] / 2,
                                                                          general_settings[Tags.DIM_VOLUME_Y_MM] / 2,
                                                                          0]))
        self.add_illumination_geometry(sp.DiskIlluminationGeometry(beam_radius_mm=2))


device = CustomDevice()

sp.simulate(pipeline, settings, device)

# visualize reconstruction results
# get simulation output
data_path = SAVE_PATH + "/" + VOLUME_NAME + ".hdf5"
file = sp.load_hdf5(data_path)
settings = sp.Settings(file["settings"])
wavelength = settings[Tags.WAVELENGTHS][0]

# get reconstruction result
absorption_reconstruction = sp.load_data_field(data_path, Tags.ITERATIVE_qPAI_RESULT, wavelength)

# get ground truth absorption coefficients
absorption_gt = sp.load_data_field(data_path, Tags.DATA_FIELD_ABSORPTION_PER_CM, wavelength)
last_fluence = np.load(SAVE_PATH + "/last_fluence_" + VOLUME_NAME + ".npy")
os.remove(SAVE_PATH + "/last_fluence_" + VOLUME_NAME + ".npy")

fluence = normalize_min_max(last_fluence)

# rescale ground truth to same dimension as reconstruction (necessary due to resampling in iterative algorithm)
scale = np.shape(absorption_reconstruction)[0] / np.shape(absorption_gt)[0]  # same as Tags.DOWNSCALE_FACTOR
absorption_gt = zoom(absorption_gt, scale, order=0, mode="nearest")
# fluence = zoom(fluence, scale, order=0, mode="nearest")


# compute reconstruction error
difference = absorption_gt - absorption_reconstruction

median_error = np.median(difference)
q3, q1 = np.percentile(difference, [75, 25])
iqr = q3 - q1

# visualize results
x_pos = int(np.shape(absorption_gt)[0] / 2)
y_pos = int(np.shape(absorption_gt)[1] / 2)

cmin = np.min(absorption_gt)
cmax = np.max(absorption_gt)

results_x_z = [absorption_gt[:, y_pos, :], absorption_reconstruction[:, y_pos, :], difference[:, y_pos, :]]
results_y_z = [absorption_gt[x_pos, :, :], absorption_reconstruction[x_pos, :, :], difference[x_pos, :, :]]

label = ["Absorption coefficients", "Reconstruction",
         "Difference"]

# plt.subplots_adjust(hspace=0.5)
# plt.suptitle("Iterative qPAI Reconstruction \n median error = " + str(np.round(median_error, 4)) +
#              "\n IQR = " + str(np.round(iqr, 4)), fontsize=10)

cmap = "jet"
fontsize = 25
fontname = "Cmr10"
for i, quantity in enumerate(results_x_z):
    # if i == 0:
    #     plt.ylabel("x-z", fontsize=10)
    # plt.title(label[i], fontsize=10)
    if i == 2:
        cmap = "Reds"
    img_plot = plt.imshow(np.rot90(quantity, -1), cmap=cmap)
    scale_bar = ScaleBar(settings[Tags.SPACING_MM]/scale, units="mm", location="lower left", font_properties={"family": fontname, "size": fontsize})
    plt.gca().add_artist(scale_bar)
    col_bar(img_plot, fontsize=fontsize, fontname=fontname)
    if i in [0, 1]:
        plt.clim(cmin, cmax)
    else:
        plt.clim(np.min(difference), np.max(difference))
    plt.tight_layout()
    if SHOW_IMAGE:
        plt.show()
        plt.close()
    else:
        plt.savefig(os.path.join(SAVE_PATH, "{}.svg".format(label[i])))
        plt.close()

img_plot = plt.imshow(np.rot90(fluence[:, y_pos, :], -1), cmap="jet")
# plt.title("Fluence")
scale_bar = ScaleBar(settings[Tags.SPACING_MM]/scale, units="mm", location="lower left", font_properties={"family": fontname, "size": fontsize})
plt.gca().add_artist(scale_bar)
col_bar(img_plot, fontsize=fontsize, fontname=fontname, ticks=[0.2, 0.4, 0.6, 0.8])
plt.tight_layout()
if SHOW_IMAGE:
    plt.show()
    plt.close()
else:
    plt.savefig(os.path.join(SAVE_PATH, "fluence.svg"))
    plt.close()
