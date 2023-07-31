# SPDX-FileCopyrightText: 2021 Intelligent Medical Systems Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa import Tags
import simpa as sp
import numpy as np
import time
import matplotlib.pyplot as plt
from utils.create_example_tissue import create_example_tissue
from utils.save_directory import get_save_path
from utils.normalizations import normalize_min_max
from utils.basic_settings import create_basic_optical_settings, create_basic_acoustic_settings, \
    create_basic_reconstruction_settings
plt.style.use("bmh")

RANDOM_SEED = 24618925
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
torch_initialize_vector = torch.zeros((10, 10), device="cuda")
fontname = "Cmr10"
fontsize = 15

REPEAT_SIMULATION = True
REPETITIONS = 100
SAVE_PATH = get_save_path("independent_execution", "independent_execution")
WAVELENGTH = 800

times = np.zeros((REPETITIONS))
images = list()

for idx in range(REPETITIONS):
    np.random.seed(RANDOM_SEED)
    VOLUME_WIDTH_HEIGHT_DIM_IN_MM = 50
    VOLUME_PLANAR_DIM_IN_MM = 50
    SPACING = 1
    path_manager = sp.PathManager()

    # Seed the numpy random configuration prior to creating the global_settings file in
    # order to ensure that the same volume
    # is generated with the same random seed every time.

    VOLUME_NAME = "ForearmScan" + str(RANDOM_SEED)
    file_path = SAVE_PATH + "/" + VOLUME_NAME + ".hdf5"

    settings = {
        # These parameters set he general propeties of the simulated volume
        Tags.RANDOM_SEED: RANDOM_SEED,
        Tags.VOLUME_NAME: VOLUME_NAME,
        Tags.SIMULATION_PATH: SAVE_PATH,
        Tags.SPACING_MM: SPACING,
        Tags.WAVELENGTHS: [WAVELENGTH],
        Tags.DIM_VOLUME_Z_MM: VOLUME_WIDTH_HEIGHT_DIM_IN_MM,
        Tags.DIM_VOLUME_X_MM: VOLUME_WIDTH_HEIGHT_DIM_IN_MM,
        Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
        Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
        Tags.GPU: True
    }

    settings = sp.Settings(settings)

    settings.set_volume_creation_settings({
        Tags.STRUCTURES: create_example_tissue(),
        Tags.SIMULATE_DEFORMED_LAYERS: True
    })
    settings.set_optical_settings(create_basic_optical_settings(path_manager))
    settings[Tags.OPTICAL_MODEL_SETTINGS][Tags.MCX_SEED] = RANDOM_SEED

    settings.set_acoustic_settings(create_basic_acoustic_settings(path_manager))

    settings.set_reconstruction_settings(create_basic_reconstruction_settings(path_manager, SPACING))

    settings["noise_time_series"] = {
        Tags.NOISE_STD: 100,
        Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
        Tags.DATA_FIELD: Tags.DATA_FIELD_TIME_SERIES_DATA
    }

    device = sp.PhotoacousticDevice(device_position_mm=np.asarray([VOLUME_WIDTH_HEIGHT_DIM_IN_MM/2,
                                                                VOLUME_PLANAR_DIM_IN_MM/2, 0]))
    device.set_detection_geometry(sp.LinearArrayDetectionGeometry(
        device_position_mm=np.asarray([VOLUME_WIDTH_HEIGHT_DIM_IN_MM/2, VOLUME_PLANAR_DIM_IN_MM/2, 0]),
        number_detector_elements=256,
        pitch_mm=0.15))
    device.add_illumination_geometry(sp.GaussianBeamIlluminationGeometry(beam_radius_mm=25))
    SIMUATION_PIPELINE = [
        sp.ModelBasedVolumeCreationAdapter(settings),
        sp.MCXAdapter(settings),
        sp.KWaveAdapter(settings),
        sp.GaussianNoise(settings, "noise_time_series"),
        sp.DelayAndSumAdapter(settings),
        sp.FieldOfViewCropping(settings)
    ]
    time_before = time.time()
    if REPEAT_SIMULATION:
        np.random.seed(RANDOM_SEED)
        sp.simulate(SIMUATION_PIPELINE, settings, device)
        recon = sp.load_data_field(file_path, Tags.DATA_FIELD_RECONSTRUCTED_DATA, WAVELENGTH)
        images.append(recon)
    needed_time = time.time() - time_before
    times[idx] = needed_time

if REPEAT_SIMULATION:
    np.savez(SAVE_PATH+"/times.npz",
             times=times)
    sp.save_hdf5({"images": np.array(images)}, file_path + ".hdf5")

times = np.load(SAVE_PATH+"/times.npz")["times"]
images = sp.load_hdf5(file_path + ".hdf5")["images"]

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


# plt.figure(figsize=(11, 4))
# plt.rc("axes", unicode_minus=False)
# plt.title("(a) Simulation times on sequentially executed tests")
# for outer_idx in range(NUM_OUTER_REPETITIONS):
#     inds = outer_idx + 0.5
#
#     parts = plt.violinplot(
#         dataset=times[outer_idx, :], positions=[inds],
#         showmeans=False, showmedians=False,
#         showextrema=False, widths=0.95)
#
#     for pc in parts['bodies']:
#         pc.set_facecolor('#AAAAAA')
#         pc.set_edgecolor('black')
#         pc.set_alpha(0.5)
#
#     quartile1, quartile3 = np.percentile(times[outer_idx, :], [25, 75])
#
#     # plt.scatter(inds, np.median(times[outer_idx, :]), marker='o', color='white', s=30, zorder=3)
#     plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
# plt.plot(np.arange(0, NUM_OUTER_REPETITIONS) + 0.5,
#          np.median(times, axis=1), lw=3, zorder=5, label="batch median run time")
# plt.plot(np.linspace(0, NUM_OUTER_REPETITIONS, NUM_OUTER_REPETITIONS*NUM_INNER_REPETITIONS),
#          np.reshape(times, (-1, )), linestyle='--', lw=2, zorder=4, label="run time")
# plt.xlabel(f"# Sequential Batch ({NUM_INNER_REPETITIONS} runs per batch)")
# plt.ylabel("Simulation run execution time [s]")
# # plt.ylim(10.8, 12.7)
# plt.legend(loc="best")
# plt.tight_layout()
# plt.savefig(SAVE_PATH + "/run_time_independence1.svg")
# plt.close()

timepoints = np.arange(0, REPETITIONS)
time = np.reshape(times, (-1, ))
mean = np.mean(time)
diff = time - mean                   # Difference between data1 and data2
md = np.mean(diff)                   # Mean of the difference
sd = np.std(diff, axis=0)            # Standard deviation of the difference

sigma_mult = 2

# plt.title("(b) Distance from mean plot")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.rc("axes", unicode_minus=False)
plt.scatter(timepoints + 1, diff, color=colors[0])
plt.axhline(md, color=colors[1], linestyle='-', label="mean difference")
plt.axhline(md + sigma_mult*sd, color=colors[2], linestyle='--', label="{} $\sigma$".format(sigma_mult))
plt.axhline(md - sigma_mult*sd, color=colors[2], linestyle='--')
plt.legend(loc=1, prop={"family": fontname, "size": fontsize})
plt.xlabel("Simulation run", fontsize=fontsize, fontname=fontname)
plt.ylabel("Distance from mean [s]", fontsize=fontsize, fontname=fontname)
plt.xticks(fontsize=fontsize, fontname=fontname)
plt.yticks(fontsize=fontsize, fontname=fontname)
ax = plt.gca()
ax.set_facecolor("white")
plt.tight_layout()
plt.savefig(SAVE_PATH + "/run_time_independence.svg")
plt.close()

for i, image in enumerate(images):
    images[i] = normalize_min_max(image)
mean_image = np.mean(images, axis=0)

deviations = list()
for image in images:
    deviations.append(np.mean(mean_image - image))
deviations = np.array(deviations)
mean = np.mean(deviations)
std = np.std(deviations)

plt.rc("axes", unicode_minus=False)
plt.scatter(timepoints + 1, deviations, color=colors[0])
plt.axhline(mean, color=colors[1], linestyle='-', label="mean difference")
plt.axhline(mean + sigma_mult*std, color=colors[2], linestyle='--', label="{} $\sigma$".format(sigma_mult))
plt.axhline(mean - sigma_mult*std, color=colors[2], linestyle='--')
plt.legend(loc=1, prop={"family": fontname, "size": fontsize})
plt.xlabel("Simulation run", fontsize=fontsize, fontname=fontname)
plt.ylabel("Distance from mean [a.u.]", fontsize=fontsize, fontname=fontname)
plt.xticks(fontsize=fontsize, fontname=fontname)
plt.yticks(fontsize=fontsize, fontname=fontname)
ax = plt.gca()
ax.yaxis.offsetText.set_fontname(fontname)
ax.set_facecolor("white")
plt.tight_layout()
plt.savefig(SAVE_PATH + "/difference_independence.svg")
plt.close()



