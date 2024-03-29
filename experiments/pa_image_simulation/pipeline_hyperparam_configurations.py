# SPDX-FileCopyrightText: 2021 Intelligent Medical Systems Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import matplotlib.pyplot as plt
from simpa import Tags
import simpa as sp
from utils.create_example_tissue import create_square_phantom
from utils.basic_settings import create_basic_reconstruction_settings, create_basic_optical_settings, \
    create_basic_acoustic_settings
import numpy as np
from skimage.transform import rescale
from utils.save_directory import get_save_path
from utils.normalizations import standardize
from matplotlib_scalebar.scalebar import ScaleBar
import os
import time

VOLUME_TRANSDUCER_DIM_IN_MM = 90
VOLUME_PLANAR_DIM_IN_MM = 20
VOLUME_HEIGHT_IN_MM = 90
RANDOM_SEED = 500
spacing_list = [0.55, 0.35, 0.15]
SHOW_IMAGE = False

hyperparam_names = {"Default": "Default\n(2D acoustic simulation\nno frequency response\n "
                               "no envelope detection\nno bandpass filtering\npressure mode)",
                    # Tags.ACOUSTIC_SIMULATION_3D: "Default\n+ 3D acoustic simulation",
                    Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: "Default\n+ bandpass filter",
                    Tags.RECONSTRUCTION_MODE_DIFFERENTIAL: "Default\n+ differential mode",
                    # Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: "Default\n+ envelope detection",
                    "Custom": "Default\n+ bandpass filter\n+ differential mode\n+ envelope detection",
                    }

hyperparam_list = ["Default",
                   # Tags.ACOUSTIC_SIMULATION_3D,
                   Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING,
                   Tags.RECONSTRUCTION_MODE_DIFFERENTIAL,
                   # Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION,
                   "Custom",
                   ]
img_arr = list()
# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = sp.PathManager()

WAVELENGTHS = [800]
SAVE_PATH = get_save_path("pa_image_simulation", "Hyperparameter_Configurations")

# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.
start_time = time.time()
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
                }
        settings = sp.Settings(general_settings)

        settings.set_volume_creation_settings({
            Tags.STRUCTURES: create_square_phantom(settings),
            Tags.US_GEL: True,

        })

        settings.set_optical_settings(create_basic_optical_settings(path_manager))

        settings.set_acoustic_settings(create_basic_acoustic_settings(path_manager))

        settings.set_reconstruction_settings(
            create_basic_reconstruction_settings(path_manager, reconstruction_spacing=min(spacing_list)))

        pa_device = sp.MSOTAcuityEcho(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM / 2,
                                                                   VOLUME_PLANAR_DIM_IN_MM / 2,
                                                                   25]),
                                      field_of_view_extent_mm=np.array([-(2 * np.sin(0.34 / 40 * 128) * 40) / 2,
                                                                        (2 * np.sin(0.34 / 40 * 128) * 40) / 2,
                                                                        0, 0, 0, 25]))
        pa_device.update_settings_for_use_of_model_based_volume_creator(settings)

        if hyperparam in [Tags.ACOUSTIC_SIMULATION_3D]:
            settings[Tags.ACOUSTIC_MODEL_SETTINGS][hyperparam] = True
        elif hyperparam in [Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION,
                            Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING]:
            settings[Tags.RECONSTRUCTION_MODEL_SETTINGS][hyperparam] = True
        elif hyperparam == Tags.RECONSTRUCTION_MODE_DIFFERENTIAL:
            settings[Tags.RECONSTRUCTION_MODEL_SETTINGS][Tags.RECONSTRUCTION_MODE] = hyperparam
        elif hyperparam == "Custom":
            settings[Tags.RECONSTRUCTION_MODEL_SETTINGS][Tags.RECONSTRUCTION_MODE] = \
                Tags.RECONSTRUCTION_MODE_DIFFERENTIAL
            settings[Tags.RECONSTRUCTION_MODEL_SETTINGS][Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION] = True
            settings[Tags.RECONSTRUCTION_MODEL_SETTINGS][Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING] = True

        SIMUATION_PIPELINE = [
            sp.ModelBasedVolumeCreationAdapter(settings),
            sp.MCXAdapter(settings),
            sp.KWaveAdapter(settings),
            sp.DelayAndSumAdapter(settings),
            sp.FieldOfViewCropping(settings),
        ]

        timer = time.time()
        sp.simulate(SIMUATION_PIPELINE, settings, pa_device)
        print("Needed", time.time()-timer, "seconds")
        # TODO global_settings[Tags.SIMPA_OUTPUT_PATH]
        print("Simulating ", RANDOM_SEED, "[Done]")

        if i == 0:
            # mua = load_data_field(SAVE_PATH + "/" + VOLUME_NAME + ".hdf5", Tags.PROPERTY_ABSORPTION_PER_CM,
            #                       WAVELENGTHS[0])
            # mua = rescale(mua, spacing / min(spacing_list), order=2, anti_aliasing=True)
            # img_arr.append(mua)

            p0 = sp.load_data_field(SAVE_PATH + "/" + VOLUME_NAME + ".hdf5", Tags.DATA_FIELD_INITIAL_PRESSURE,
                                    WAVELENGTHS[0])
            p0 = rescale(p0, spacing/min(spacing_list), order=2, anti_aliasing=True)
            norm_p0, _, _ = standardize(p0, log=False)
            img_arr.append(norm_p0)

        recon = sp.load_data_field(SAVE_PATH + "/" + VOLUME_NAME + ".hdf5", Tags.DATA_FIELD_RECONSTRUCTED_DATA,
                                   WAVELENGTHS[0])
        norm_recon, _, _ = standardize(recon, log=False)
        img_arr.append(norm_recon)
end_time = time.time() - start_time
with open(os.path.join(SAVE_PATH, "run_time.txt"), "w+") as out_file:
    out_file.write("{:.2f} s".format(end_time))

from mpl_toolkits.axes_grid1 import ImageGrid
# hyperparam_list.insert(0, "Initial pressure\n(simulated ground truth)")
hyperparam_list.insert(0, "Initial pressure")
# hyperparam_list.insert(0, "Absorption")
for h, hyperparam in enumerate(hyperparam_list):
    fig = plt.figure(figsize=(3, 5))
    img_grid = ImageGrid(fig, rect=111, nrows_ncols=(len(spacing_list), 1), axes_pad=0.05)
    for i, gridax in enumerate(img_grid):
        img = np.rot90(img_arr[h + i*len(hyperparam_list)], -1)[:, 75:-75]
        im = gridax.imshow(img, vmin=-3, vmax=3)
        # if i < len(hyperparam_list):
        #     if i < 1:
        #         label = hyperparam_list[i][0] if isinstance(hyperparam_list[i], tuple) else hyperparam_list[i]
        #     else:
        #         # tag_label = hyperparam_list[i][0] if isinstance(hyperparam_list[i], tuple) else hyperparam_list[i]
        #         label = hyperparam_names[hyperparam_list[i]]
        #     gridax.set_title(label)
        if h == 0:
            gridax.text(x=-23, y=160, s="${\Delta}$x =" + str(spacing_list[i]) + " mm", rotation=90, fontsize=14,
                        fontname="Cmr10")
            scale_bar = ScaleBar(settings[Tags.SPACING_MM], units="mm", location="lower left",
                                 font_properties={"family": "Cmr10", "size": 14})
            gridax.add_artist(scale_bar)
        gridax.axis('off')

    # cbar = img_grid.cbar_axes[0].colorbar(im)
    # cbar.set_label_text("Signal intensity [normalised units]")
    if SHOW_IMAGE:
        plt.show()
        plt.close()
    else:
        plt.savefig(SAVE_PATH + "/{}.svg".format(hyperparam[0] if isinstance(hyperparam, tuple) else hyperparam))
        plt.close()
        fontsize = 30
        fontname = "Cmr10"
        fig = plt.figure()
        plt.rc("axes", unicode_minus=False)
        ax = fig.add_axes([0.01, 0.02, 0.025, 0.95])
        import matplotlib as mpl

        cbar = mpl.colorbar.ColorbarBase(ax, norm=mpl.colors.Normalize(-3, 3), ticks=[-3, -2, -1, 0, 1, 2, 3])
        for l in cbar.ax.yaxis.get_ticklabels():
            l.set_family(fontname)
        cbar.ax.tick_params(labelsize=fontsize/2)
        plt.savefig(SAVE_PATH + "/cbar_hyperparam.svg", bbox_inches="tight")
