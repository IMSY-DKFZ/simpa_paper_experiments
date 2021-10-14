from simpa.utils import Tags
from simpa.utils.path_manager import PathManager
from simpa.io_handling import load_data_field, save_hdf5, load_hdf5
from simpa.core.simulation import simulate
from simpa.simulation_components import VolumeCreationModelModelBasedAdapter, OpticalForwardModelMcxAdapter, \
    AcousticForwardModelKWaveAdapter, ImageReconstructionModuleDelayAndSumAdapter, FieldOfViewCroppingProcessingComponent
from simpa.utils import Settings, TISSUE_LIBRARY
from simpa.core.device_digital_twins import PhotoacousticDevice, GaussianBeamIlluminationGeometry, LinearArrayDetectionGeometry
from simpa.core.processing_components import GaussianNoiseProcessingComponent
import numpy as np
import time
import matplotlib.pyplot as plt
from simpa.utils.libraries.structure_library import define_horizontal_layer_structure_settings, \
    define_vessel_structure_settings, define_circular_tubular_structure_settings, define_background_structure_settings
from utils.save_directory import get_save_path
from utils.normalizations import normalize_min_max
plt.style.use("bmh")

RANDOM_SEED = 24618925
np.random.seed(RANDOM_SEED)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
torch_initialize_vector = torch.zeros((10, 10), device="cuda")
fontname = "Cmr10"
fontsize = 15

REPEAT_SIMULATION = False
NUM_INNER_REPETITIONS = 10
NUM_OUTER_REPETITIONS = 10
SAVE_PATH = get_save_path("independent_execution", "independent_execution")
WAVELENGTH = 800

times = np.zeros((NUM_OUTER_REPETITIONS, NUM_INNER_REPETITIONS))
images = list()

for outer_idx in range(NUM_OUTER_REPETITIONS):
    for inner_idx in range(NUM_INNER_REPETITIONS):
        VOLUME_WIDTH_HEIGHT_DIM_IN_MM = 50
        VOLUME_PLANAR_DIM_IN_MM = 50
        SPACING = 1
        path_manager = PathManager()

        def create_example_tissue():
            tissue_dict = Settings()
            tissue_dict[Tags.BACKGROUND] = define_background_structure_settings(molecular_composition=TISSUE_LIBRARY.ultrasound_gel())
            tissue_dict["epidermis"] = define_horizontal_layer_structure_settings(z_start_mm=2+0, thickness_mm=2,
                                                                                  adhere_to_deformation=True,
                                                                                  molecular_composition=TISSUE_LIBRARY.constant(0.5, 10, 0.9),
                                                                                  consider_partial_volume=True)
            tissue_dict["dermis"] = define_horizontal_layer_structure_settings(z_start_mm=2+2, thickness_mm=9,
                                                                               adhere_to_deformation=True,
                                                                               molecular_composition=TISSUE_LIBRARY.constant(0.05, 10, 0.9),
                                                                               consider_partial_volume=True)
            tissue_dict["fat"] = define_horizontal_layer_structure_settings(z_start_mm=2+11, thickness_mm=4,
                                                                            adhere_to_deformation=True,
                                                                            molecular_composition=TISSUE_LIBRARY.constant(0.05, 10, 0.9),
                                                                            consider_partial_volume=True)
            tissue_dict["vessel_1"] = define_vessel_structure_settings(vessel_start_mm=[25, 0, 2+17],
                                                                       vessel_direction_mm=[-0.05, 1, 0],
                                                                       radius_mm=2, bifurcation_length_mm=100,
                                                                       curvature_factor=0.01,
                                                                       molecular_composition=TISSUE_LIBRARY.constant(1.3, 10, 0.9),
                                                                       consider_partial_volume=True)
            tissue_dict["vessel_2"] = define_vessel_structure_settings(vessel_start_mm=[5, 0, 2+17],
                                                                       vessel_direction_mm=[0, 1, 0],
                                                                       radius_mm=1.5, bifurcation_length_mm=100,
                                                                       curvature_factor=0.01,
                                                                       molecular_composition=TISSUE_LIBRARY.constant(1.3, 10, 0.9),
                                                                       consider_partial_volume=True)
            tissue_dict["vessel_3"] = define_vessel_structure_settings(vessel_start_mm=[45, 0, 2+19],
                                                                       vessel_direction_mm=[0.05, 1, 0],
                                                                       radius_mm=1.5, bifurcation_length_mm=100,
                                                                       curvature_factor=0.01,
                                                                       molecular_composition=TISSUE_LIBRARY.constant(1.3, 10, 0.9),
                                                                       consider_partial_volume=True)
            tissue_dict["vessel_4"] = define_vessel_structure_settings(vessel_start_mm=[25, 0, 2+35],
                                                                       vessel_direction_mm=[0.05, 1, 0],
                                                                       radius_mm=6, bifurcation_length_mm=15,
                                                                       curvature_factor=0.1,
                                                                       molecular_composition=TISSUE_LIBRARY.constant(1.3, 10, 0.9),
                                                                       consider_partial_volume=True)
            tissue_dict["bone"] = define_circular_tubular_structure_settings(tube_start_mm=[5, 0, 45], tube_end_mm=[5, 50, 45],
                                                                             radius_mm=15,
                                                                             molecular_composition=TISSUE_LIBRARY.constant(1.3, 10, 0.9),
                                                                             consider_partial_volume=True)
            return tissue_dict

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

        settings = Settings(settings)

        settings.set_volume_creation_settings({
            Tags.STRUCTURES: create_example_tissue(),
            Tags.SIMULATE_DEFORMED_LAYERS: True
        })
        settings.set_optical_settings({
            Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
            Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
            Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
            Tags.MCX_ASSUMED_ANISOTROPY: 0.9,
            Tags.MCX_SEED: RANDOM_SEED
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
            Tags.GPU: True
        })
        settings.set_reconstruction_settings({
            Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: False,
            Tags.PROPERTY_ALPHA_POWER: 0.00,
            Tags.TUKEY_WINDOW_ALPHA: 0.5,
            Tags.BANDPASS_CUTOFF_LOWPASS: int(8e6),
            Tags.BANDPASS_CUTOFF_HIGHPASS: int(0.1e4),
            Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: False,
            Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
            Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
            Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE,
            Tags.SPACING_MM: SPACING,
            Tags.GPU: True
        })
        settings["noise_time_series"] = {
            Tags.NOISE_STD: 100,
            Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
            Tags.DATA_FIELD: Tags.TIME_SERIES_DATA
        }

        device = PhotoacousticDevice(device_position_mm=np.asarray([VOLUME_WIDTH_HEIGHT_DIM_IN_MM/2,
                                                                    VOLUME_PLANAR_DIM_IN_MM/2, 0]))
        device.set_detection_geometry(LinearArrayDetectionGeometry(
            device_position_mm=np.asarray([VOLUME_WIDTH_HEIGHT_DIM_IN_MM/2, VOLUME_PLANAR_DIM_IN_MM/2, 0]),
            number_detector_elements=256,
            pitch_mm=0.15))
        device.add_illumination_geometry(GaussianBeamIlluminationGeometry(beam_radius_mm=25))
        SIMUATION_PIPELINE = [
            VolumeCreationModelModelBasedAdapter(settings),
            OpticalForwardModelMcxAdapter(settings),
            AcousticForwardModelKWaveAdapter(settings),
            GaussianNoiseProcessingComponent(settings, "noise_time_series"),
            ImageReconstructionModuleDelayAndSumAdapter(settings),
            FieldOfViewCroppingProcessingComponent(settings)
        ]

        time_before = time.time()
        if REPEAT_SIMULATION:
            simulate(SIMUATION_PIPELINE, settings, device)
            recon = load_data_field(file_path, Tags.OPTICAL_MODEL_INITIAL_PRESSURE, WAVELENGTH)
            images.append(recon)
        needed_time = time.time() - time_before
        times[outer_idx, inner_idx] = needed_time

if REPEAT_SIMULATION:
    np.savez(SAVE_PATH+"/times.npz",
             times=times)
    save_hdf5({"images": np.array(images)}, file_path + ".hdf5")

times = np.load(SAVE_PATH+"/times.npz")["times"]
images = load_hdf5(file_path + ".hdf5")["images"]

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

timepoints = np.arange(0, NUM_OUTER_REPETITIONS*NUM_INNER_REPETITIONS)
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
plt.tight_layout()
plt.savefig(SAVE_PATH + "/difference_independence.svg")
plt.close()



