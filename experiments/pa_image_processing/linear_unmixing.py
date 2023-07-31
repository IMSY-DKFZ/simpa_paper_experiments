# SPDX-FileCopyrightText: 2021 Intelligent Medical Systems Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa import Tags
import simpa as sp
import numpy as np
import matplotlib.pyplot as plt
from utils.save_directory import get_save_path
from utils.basic_settings import create_basic_acoustic_settings, create_basic_reconstruction_settings
from visualization.colorbar import col_bar
from utils.create_example_tissue import create_linear_unmixing_phantom
from matplotlib_scalebar.scalebar import ScaleBar
import time

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = sp.PathManager()
SAVE_PATH = get_save_path("pa_image_processing", "Linear_Unmixing")

# set global params characterizing the simulated volume
VOLUME_TRANSDUCER_DIM_IN_MM = 75
VOLUME_PLANAR_DIM_IN_MM = 20
VOLUME_HEIGHT_IN_MM = 25
SPACING = 0.2
RANDOM_SEED = 471
VOLUME_NAME = "LinearUnmixing_" + str(RANDOM_SEED)

# since we want to perform linear unmixing, the simulation pipeline should be execute for at least two wavelengths
# WAVELENGTHS = list(np.arange(750, 850, 10))
WAVELENGTHS = [750, 800, 850]


# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume is generated with the same random seed every time.
np.random.seed(RANDOM_SEED)

# Initialize global settings and prepare for simulation pipeline including
# volume creation and optical forward simulation.
general_settings = {
    # These parameters set the general properties of the simulated volume
    Tags.RANDOM_SEED: RANDOM_SEED,
    Tags.VOLUME_NAME: VOLUME_NAME,
    Tags.SIMULATION_PATH: SAVE_PATH,
    Tags.SPACING_MM: SPACING,
    Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
    Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
    Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
    Tags.WAVELENGTHS: WAVELENGTHS,
    Tags.GPU: True,

}
settings = sp.Settings(general_settings)
settings.set_volume_creation_settings({
    Tags.SIMULATE_DEFORMED_LAYERS: True,
    Tags.STRUCTURES: create_linear_unmixing_phantom(settings)
})
settings.set_optical_settings({
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
    Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
    Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
    Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50
})

settings["noise_initial_pressure"] = {
            Tags.NOISE_MEAN: 1,
            Tags.NOISE_STD: 0.01,
            Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
            Tags.DATA_FIELD: Tags.DATA_FIELD_INITIAL_PRESSURE,
            Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
        }

settings.set_acoustic_settings(create_basic_acoustic_settings(path_manager))

recon_settings = create_basic_reconstruction_settings(path_manager, reconstruction_spacing=SPACING)
recon_settings[Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION] = True
recon_settings[Tags.RECONSTRUCTION_MODE] = Tags.RECONSTRUCTION_MODE_DIFFERENTIAL
settings.set_reconstruction_settings(recon_settings)

# Set component settings for linear unmixing.
# In this example we are only interested in the chromophore concentration of oxy- and deoxyhemoglobin and the
# resulting blood oxygen saturation. We want to perform the algorithm using all three wavelengths defined above.
# Please take a look at the component for more information.
wavelengths_to_unmix = [750, 850]
unmixing_data_field = Tags.DATA_FIELD_RECONSTRUCTED_DATA
settings["linear_unmixing"] = {
    Tags.DATA_FIELD: unmixing_data_field,
    Tags.WAVELENGTHS: wavelengths_to_unmix,
    Tags.LINEAR_UNMIXING_COMPUTE_SO2: True,
    Tags.LINEAR_UNMIXING_SPECTRA: sp.get_simpa_internal_absorption_spectra_by_names(
        [Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN, Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN]),
    Tags.SIGNAL_THRESHOLD: 0.00,
}

# Get device for simulation
device = sp.MSOTAcuityEcho(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                        VOLUME_PLANAR_DIM_IN_MM/2,
                                                        0]),
                           field_of_view_extent_mm=np.asarray([-20,
                                                               20,
                                                               0, 0, 0, 20])
                           )
device.update_settings_for_use_of_model_based_volume_creator(settings)

# Run simulation pipeline for all wavelengths in Tag.WAVELENGTHS
pipeline = [
    sp.ModelBasedVolumeCreationAdapter(settings),
    sp.MCXAdapter(settings),
    sp.GaussianNoise(settings, "noise_initial_pressure"),
    sp.KWaveAdapter(settings),
    sp.DelayAndSumAdapter(settings),
    sp.FieldOfViewCropping(settings)
]

start_time = time.time()
sp.simulate(pipeline, settings, device)

# Run linear unmixing component with above specified settings.
file_path = SAVE_PATH + "/" + VOLUME_NAME + ".hdf5"
settings[Tags.SIMPA_OUTPUT_PATH] = file_path
sp.LinearUnmixing(settings, "linear_unmixing").run()
end_time = time.time() - start_time
with open(os.path.join(SAVE_PATH, "run_time.txt"), "w+") as out_file:
    out_file.write("{:.2f} s".format(end_time))

# Load linear unmixing result (blood oxygen saturation) and reference absorption for first wavelength.
lu_results = sp.load_data_field(file_path, Tags.LINEAR_UNMIXING_RESULT)
sO2 = lu_results["sO2"] * 100

mua = sp.load_data_field(file_path, Tags.DATA_FIELD_ABSORPTION_PER_CM, wavelength=WAVELENGTHS[0])
p0 = sp.load_data_field(file_path, Tags.DATA_FIELD_INITIAL_PRESSURE, wavelength=WAVELENGTHS[0])
gt_oxy = sp.load_data_field(file_path, Tags.DATA_FIELD_OXYGENATION, wavelength=WAVELENGTHS[0]) * 100
DATA_FIELD_RECONSTRUCTED_DATA = sp.load_data_field(file_path, Tags.DATA_FIELD_RECONSTRUCTED_DATA, wavelength=WAVELENGTHS[0])

# PLOT_SPECTRA = True
# if PLOT_SPECTRA:
#     mua_spectrum, p0_spectrum = list(), list()
#     example_coordinate = [114, 17]
#     for wavelength in WAVELENGTHS:
#         mua_spectrum.append(load_data_field(file_path, Tags.PROPERTY_ABSORPTION_PER_CM,
#                                             wavelength=wavelength)[example_coordinate[0], example_coordinate[1]])
#         p0_spectrum.append(load_data_field(file_path, unmixing_data_field,
#                                            wavelength=wavelength)[example_coordinate[0], example_coordinate[1]])
#
#     mua_spectrum = np.array(mua_spectrum) / max(mua_spectrum)
#     p0_spectrum = np.array(p0_spectrum) / max(p0_spectrum)
#     plt.plot(WAVELENGTHS, mua_spectrum, label="Absorption")
#     plt.plot(WAVELENGTHS, p0_spectrum, label="Initial Pressure")
#     plt.legend()
#     plt.show()
#     plt.close()


# Visualize linear unmixing result
data_shape = mua.shape
if len(data_shape) == 3:
    y_dim = int(data_shape[1] / 2)
    p0 = p0[:, y_dim, :]
    mua = mua[:, y_dim, :]
    gt_oxy = gt_oxy[:, y_dim, :]

SHOW_IMAGE = False
fontsize = 30
fontname = "Cmr10"
# plt.title("Reconstructed PA\nimage @750nm [a.u.]")
recon = plt.imshow(np.rot90(DATA_FIELD_RECONSTRUCTED_DATA/np.max(DATA_FIELD_RECONSTRUCTED_DATA), -1))
scale_bar = ScaleBar(settings[Tags.SPACING_MM], units="mm", location="lower center", font_properties={"family": "Cmr10", "size": fontsize})
plt.gca().add_artist(scale_bar)
col_bar(recon, fontsize=fontsize, fontname=fontname, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.clim(0, 1)
plt.tight_layout()
if SHOW_IMAGE:
    plt.show()
    plt.close()
else:
    plt.savefig(os.path.join(SAVE_PATH, "recon.svg"))
    plt.close()
# plt.title("Ground truth blood\noxygen saturation (gt) [%]")
gt_im = plt.imshow(np.rot90(gt_oxy, -1))
scale_bar = ScaleBar(settings[Tags.SPACING_MM], units="mm", location="lower center", font_properties={"family": "Cmr10", "size": fontsize})
plt.gca().add_artist(scale_bar)
col_bar(gt_im, fontsize=fontsize, fontname=fontname, ticks=[0, 20, 40, 60, 80, 100])
plt.tight_layout()
if SHOW_IMAGE:
    plt.show()
    plt.close()
else:
    plt.savefig(os.path.join(SAVE_PATH, "ground_truth.svg"))
    plt.close()
# plt.title("Estimated blood\noxygen saturation (est) [%]")
plt.imshow(np.rot90(sO2, -1))
scale_bar = ScaleBar(settings[Tags.SPACING_MM], units="mm", location="lower center", font_properties={"family": "Cmr10", "size": fontsize})
plt.gca().add_artist(scale_bar)
col_bar(gt_im, fontsize=fontsize, fontname=fontname, ticks=[0, 20, 40, 60, 80, 100])
plt.clim(np.min(gt_oxy), np.max(gt_oxy))
plt.tight_layout()
if SHOW_IMAGE:
    plt.show()
    plt.close()
else:
    plt.savefig(os.path.join(SAVE_PATH, "LU_result.svg"))
    plt.close()
# plt.title("Abosolute Error |est-gt| [%]")
recon = plt.imshow(np.rot90(np.abs(sO2 - gt_oxy), -1), cmap="Reds", vmin=0, vmax=50)
scale_bar = ScaleBar(settings[Tags.SPACING_MM], units="mm", location="lower center", font_properties={"family": "Cmr10", "size": fontsize})
plt.gca().add_artist(scale_bar)
col_bar(recon, fontsize=fontsize, fontname=fontname, ticks=[0, 10, 20, 30, 40, 50])
plt.tight_layout()
if SHOW_IMAGE:
    plt.show()
    plt.close()
else:
    plt.savefig(os.path.join(SAVE_PATH, "abs_error.svg"))
    plt.close()
