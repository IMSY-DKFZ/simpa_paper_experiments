from simpa.utils import Tags, SegmentationClasses
from simpa.utils.path_manager import PathManager

from simpa.core.simulation import simulate
from simpa.simulation_components import VolumeCreationModelModelBasedAdapter, OpticalForwardModelMcxAdapter, \
    AcousticForwardModelKWaveAdapter, ImageReconstructionModuleDelayAndSumAdapter, \
    FieldOfViewCroppingProcessingComponent
from simpa.utils import Settings, TISSUE_LIBRARY
from simpa.core.device_digital_twins import PhotoacousticDevice, GaussianBeamIlluminationGeometry, \
    LinearArrayDetectionGeometry
from simpa.core.processing_components import GaussianNoiseProcessingComponent
from simpa.io_handling import load_data_field
import numpy as np
import inspect
import matplotlib.pyplot as plt
from simpa.utils.libraries.structure_library import define_horizontal_layer_structure_settings, \
    define_vessel_structure_settings, define_circular_tubular_structure_settings, define_background_structure_settings
from utils.save_directory import get_save_path
from simpa.visualisation.matplotlib_data_visualisation import visualise_data


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VOLUME_WIDTH_HEIGHT_DIM_IN_MM = 50
VOLUME_PLANAR_DIM_IN_MM = 50
SPACING = 0.3
RANDOM_SEED = 24618925

path_manager = PathManager()
SAVE_PATH = get_save_path("full_simulation", "forearm")


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


np.random.seed(RANDOM_SEED)
VOLUME_NAME = "ForearmScan" + str(RANDOM_SEED)
file_path = SAVE_PATH + "/" + VOLUME_NAME + ".hdf5"

settings = {
    # These parameters set he general propeties of the simulated volume
    Tags.RANDOM_SEED: RANDOM_SEED,
    Tags.VOLUME_NAME: VOLUME_NAME,
    Tags.SIMULATION_PATH: SAVE_PATH,
    Tags.SPACING_MM: SPACING,
    Tags.WAVELENGTHS: [800],
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
device.set_detection_geometry(LinearArrayDetectionGeometry(device_position_mm=np.asarray([VOLUME_WIDTH_HEIGHT_DIM_IN_MM/2,
                                                            VOLUME_PLANAR_DIM_IN_MM/2, 0]),
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

simulate(SIMUATION_PIPELINE, settings, device)
wavelength = settings[Tags.WAVELENGTHS][0]

segmentation_mask = load_data_field(file_path=file_path,
                                    wavelength=wavelength,
                                    data_field=Tags.PROPERTY_SEGMENTATION)

reco = np.rot90(load_data_field(file_path, wavelength=wavelength, data_field=Tags.RECONSTRUCTED_DATA), -1)
time_series = np.rot90(load_data_field(file_path, wavelength=wavelength, data_field=Tags.TIME_SERIES_DATA), -1)
initial_pressure = np.rot90(load_data_field(file_path, wavelength=wavelength, data_field=Tags.OPTICAL_MODEL_INITIAL_PRESSURE), -1)

plt.figure(figsize=(7, 3))
plt.subplot(1, 3, 1)
plt.axis('off')
plt.imshow(initial_pressure)
plt.subplot(1, 3, 2)
plt.axis('off')
plt.imshow(time_series, aspect=0.18)
plt.subplot(1, 3, 3)
plt.axis('off')
plt.imshow(reco)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "result.svg"), dpi=300)
plt.close()

# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection='3d')
# ax.voxels(segmentation_mask==SegmentationClasses.EPIDERMIS, shade=True, facecolors="brown", alpha=0.45)
# ax.voxels(segmentation_mask==SegmentationClasses.DERMIS, shade=True, facecolors="pink", alpha=0.45)
# ax.voxels(segmentation_mask==SegmentationClasses.FAT, shade=True, facecolors="yellow", alpha=0.45)
# ax.voxels(segmentation_mask==SegmentationClasses.BLOOD, shade=True, facecolors="red", alpha=0.55)
# ax.voxels(segmentation_mask==SegmentationClasses.BONE, shade=True, facecolors="antiquewhite", alpha=0.55)
# ax.set_aspect('auto')
# ax.set_zlim(50, 0)
# ax.set_zlabel("Depth [mm]")
# ax.set_xlabel("x width [mm]")
# ax.set_ylabel("y width [mm]")
# ax.view_init(elev=10., azim=-45)
# plt.savefig(os.path.join(SAVE_PATH, "forearm.svg"), dpi=300)
# plt.show()