from simpa.utils import Tags


def create_basic_optical_settings(path_manager):
    optical_settings = {
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
        Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
        Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
        Tags.MCX_ASSUMED_ANISOTROPY: 0.9,
    }

    return optical_settings


def create_basic_acoustic_settings(path_manager):

    acoustic_settings = {
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
        Tags.INITIAL_PRESSURE_SMOOTHING: True,
    }
    return acoustic_settings


def create_basic_reconstruction_settings(path_manager, reconstruction_spacing: float = 0.2):
    reconstruction_settings = {
        Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: False,
        Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
        # Tags.ACOUSTIC_SIMULATION_3D: True,
        Tags.PROPERTY_ALPHA_POWER: 0.0,
        Tags.TUKEY_WINDOW_ALPHA: 0.5,
        Tags.BANDPASS_CUTOFF_LOWPASS: int(8e6),
        Tags.BANDPASS_CUTOFF_HIGHPASS: int(0.1e4),
        Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: False,
        Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
        Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
        Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE,
        Tags.SENSOR_RECORD: "p",
        Tags.PMLInside: False,
        Tags.PMLSize: [31, 32],
        Tags.PMLAlpha: 1.5,
        Tags.PlotPML: False,
        Tags.RECORDMOVIE: False,
        Tags.MOVIENAME: "visualization_log",
        Tags.ACOUSTIC_LOG_SCALE: True,
        Tags.PROPERTY_SPEED_OF_SOUND: 1540,
        Tags.SPACING_MM: reconstruction_spacing,
        Tags.MODEL_SENSOR_FREQUENCY_RESPONSE: False
    }
    return reconstruction_settings
