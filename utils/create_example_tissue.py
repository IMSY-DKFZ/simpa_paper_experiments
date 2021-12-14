from simpa import Tags
import simpa as sp
import numpy as np
import nrrd
from scipy.ndimage import zoom


def create_example_tissue():
    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = sp.define_background_structure_settings(
        molecular_composition=sp.TISSUE_LIBRARY.ultrasound_gel())
    tissue_dict["epidermis"] = sp.define_horizontal_layer_structure_settings(z_start_mm=2 + 0, thickness_mm=2,
                                                                             adhere_to_deformation=True,
                                                                             molecular_composition=sp.TISSUE_LIBRARY.
                                                                             constant(0.5, 10, 0.9),
                                                                             consider_partial_volume=True)
    tissue_dict["dermis"] = sp.define_horizontal_layer_structure_settings(z_start_mm=2 + 2, thickness_mm=9,
                                                                          adhere_to_deformation=True,
                                                                          molecular_composition=sp.TISSUE_LIBRARY.
                                                                          constant(0.05, 10, 0.9),
                                                                          consider_partial_volume=True)
    tissue_dict["fat"] = sp.define_horizontal_layer_structure_settings(z_start_mm=2 + 11, thickness_mm=4,
                                                                       adhere_to_deformation=True,
                                                                       molecular_composition=sp.TISSUE_LIBRARY.
                                                                       constant(0.05, 10, 0.9),
                                                                       consider_partial_volume=True)
    tissue_dict["vessel_1"] = sp.define_vessel_structure_settings(vessel_start_mm=[25, 0, 2 + 17],
                                                                  vessel_direction_mm=[-0.05, 1, 0],
                                                                  radius_mm=2, bifurcation_length_mm=100,
                                                                  curvature_factor=0.01,
                                                                  molecular_composition=sp.TISSUE_LIBRARY.
                                                                  constant(1.3, 10, 0.9),
                                                                  consider_partial_volume=True)
    tissue_dict["vessel_2"] = sp.define_vessel_structure_settings(vessel_start_mm=[5, 0, 2 + 17],
                                                                  vessel_direction_mm=[0, 1, 0],
                                                                  radius_mm=1.5, bifurcation_length_mm=100,
                                                                  curvature_factor=0.01,
                                                                  molecular_composition=sp.TISSUE_LIBRARY.
                                                                  constant(1.3, 10, 0.9),
                                                                  consider_partial_volume=True)
    tissue_dict["vessel_3"] = sp.define_vessel_structure_settings(vessel_start_mm=[45, 0, 2 + 19],
                                                                  vessel_direction_mm=[0.05, 1, 0],
                                                                  radius_mm=1.5, bifurcation_length_mm=100,
                                                                  curvature_factor=0.01,
                                                                  molecular_composition=sp.TISSUE_LIBRARY.
                                                                  constant(1.3, 10, 0.9),
                                                                  consider_partial_volume=True)
    tissue_dict["vessel_4"] = sp.define_vessel_structure_settings(vessel_start_mm=[25, 0, 2 + 35],
                                                                  vessel_direction_mm=[0.05, 1, 0],
                                                                  radius_mm=6, bifurcation_length_mm=15,
                                                                  curvature_factor=0.1,
                                                                  molecular_composition=sp.TISSUE_LIBRARY.
                                                                  constant(1.3, 10, 0.9),
                                                                  consider_partial_volume=True)
    tissue_dict["bone"] = sp.define_circular_tubular_structure_settings(tube_start_mm=[5, 0, 45],
                                                                        tube_end_mm=[5, 50, 45],
                                                                        radius_mm=15,
                                                                        molecular_composition=sp.TISSUE_LIBRARY.
                                                                        constant(1.3, 10, 0.9),
                                                                        consider_partial_volume=True)
    return tissue_dict


def create_square_phantom(settings):
    x_dim = settings[Tags.DIM_VOLUME_X_MM]
    y_dim = settings[Tags.DIM_VOLUME_Y_MM]
    z_dim = settings[Tags.DIM_VOLUME_Z_MM]

    background_dictionary = sp.Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-10, 1e-10, 1.0)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    water_dictionary = sp.define_horizontal_layer_structure_settings(
        priority=2,
        molecular_composition=sp.TISSUE_LIBRARY.heavy_water(),
        z_start_mm=0,
        thickness_mm=z_dim,
        consider_partial_volume=True,
        adhere_to_deformation=False
    )

    epidermis_dictionary = sp.define_rectangular_cuboid_structure_settings(
        priority=3,
        molecular_composition=sp.TISSUE_LIBRARY.epidermis(),
        start_mm=[28, 0, 28],
        extent_mm=[34, 20, 34],
        consider_partial_volume=True,
        adhere_to_deformation=False
    )

    muscle_dictionary = sp.define_rectangular_cuboid_structure_settings(
        priority=5,
        molecular_composition=sp.TISSUE_LIBRARY.constant(0.05, 100, 0.9),
        start_mm=[28.1, 0, 28.1],
        extent_mm=[33.8, 19.8, 33.8],
        consider_partial_volume=True,
        adhere_to_deformation=False
    )

    vessel_1_dictionary = sp.define_circular_tubular_structure_settings(
        priority=8,
        molecular_composition=sp.TISSUE_LIBRARY.blood(),
        tube_start_mm=[x_dim/2 + 10, 0, z_dim/2 + 10],
        tube_end_mm=[x_dim/2 + 10, y_dim, z_dim/2 + 10],
        radius_mm=3,
        consider_partial_volume=True,
        adhere_to_deformation=False
    )

    vessel_2_dictionary = sp.define_elliptical_tubular_structure_settings(
        priority=8,
        molecular_composition=sp.TISSUE_LIBRARY.blood(),
        tube_start_mm=[x_dim/2 - 10, 0, z_dim/2 - 10],
        tube_end_mm=[x_dim/2 - 10, y_dim, z_dim/2 - 10],
        radius_mm=3,
        eccentricity=0.8,
        consider_partial_volume=True,
        adhere_to_deformation=False
    )

    inclusion_1_dictionary = sp.define_rectangular_cuboid_structure_settings(
        priority=8,
        molecular_composition=sp.TISSUE_LIBRARY.blood(),
        start_mm=[30, 0, 50],
        extent_mm=[5, 20, 8],
        consider_partial_volume=False,
        adhere_to_deformation=False
    )

    inclusion_2_dictionary = sp.define_parallelepiped_structure_settings(
        start_mm=[50, 0, 30],
        edge_a_mm=[-8, 0, 5],
        edge_b_mm=[0, 20, 0],
        edge_c_mm=[2, 0, 6],
        molecular_composition=sp.TISSUE_LIBRARY.blood()
    )

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    tissue_dict["water"] = water_dictionary
    tissue_dict["muscle"] = muscle_dictionary
    tissue_dict["epidermis"] = epidermis_dictionary
    tissue_dict["vessel_1"] = vessel_1_dictionary
    tissue_dict["vessel_2"] = vessel_2_dictionary
    tissue_dict["inclusion_1"] = inclusion_1_dictionary
    tissue_dict["inclusion_2"] = inclusion_2_dictionary
    return tissue_dict


def create_qPAI_phantom(settings):
    x_dim = settings[Tags.DIM_VOLUME_X_MM]
    y_dim = settings[Tags.DIM_VOLUME_Y_MM]
    z_dim = settings[Tags.DIM_VOLUME_Z_MM]

    background_dictionary = sp.Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-10, 1e-10, 1.0)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND
    background_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = False

    scattering = 100
    assumed_anisotropy = 0.8

    water_dictionary = sp.define_horizontal_layer_structure_settings(
        molecular_composition=sp.TISSUE_LIBRARY.constant(mua=0.1, mus=scattering, g=assumed_anisotropy),
        priority=1,
        z_start_mm=0,
        thickness_mm=z_dim,
        consider_partial_volume=False
    )

    vessel_1_dictionary = sp.define_circular_tubular_structure_settings(
        priority=5,
        tube_start_mm=[x_dim/2, 0, 1.5],
        tube_end_mm=[x_dim/2, y_dim, 1.5],
        radius_mm=0.5,
        molecular_composition=sp.TISSUE_LIBRARY.constant(mua=3, mus=scattering, g=assumed_anisotropy),
        consider_partial_volume=False
    )

    vessel_2_dictionary = sp.define_circular_tubular_structure_settings(
        priority=5,
        tube_start_mm=[x_dim-3, 0, 4.75],
        tube_end_mm=[x_dim-3, y_dim, 4.75],
        radius_mm=1,
        molecular_composition=sp.TISSUE_LIBRARY.constant(mua=1, mus=scattering, g=assumed_anisotropy),
        consider_partial_volume=False
    )

    inclusion_dictionary = sp.define_rectangular_cuboid_structure_settings(
        priority=5,
        start_mm=[3, 0, 6.5],
        extent_mm=[2, y_dim, 3],
        molecular_composition=sp.TISSUE_LIBRARY.constant(mua=2, mus=scattering, g=assumed_anisotropy),
        consider_partial_volume=False
    )

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    tissue_dict["water"] = water_dictionary
    tissue_dict["vessel_1"] = vessel_1_dictionary
    tissue_dict["vessel_2"] = vessel_2_dictionary
    tissue_dict["inclusion"] = inclusion_dictionary
    return tissue_dict


def create_linear_unmixing_phantom(settings):
    x_dim = settings[Tags.DIM_VOLUME_X_MM]
    y_dim = settings[Tags.DIM_VOLUME_Y_MM]
    z_dim = settings[Tags.DIM_VOLUME_Z_MM]

    background_dictionary = sp.Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-4, 1e-4, 0.9)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    muscle_dictionary = sp.define_horizontal_layer_structure_settings(
        priority=1,
        molecular_composition=sp.TISSUE_LIBRARY.constant(0.5, 10, 0.9),
        z_start_mm=0,
        thickness_mm=z_dim,
        consider_partial_volume=True,
        adhere_to_deformation=True
    )

    vessel_1_dictionary = sp.define_circular_tubular_structure_settings(
        priority=3,
        molecular_composition=sp.TISSUE_LIBRARY.blood(oxygenation=1),
        tube_start_mm=[x_dim/2 - 7, 0, 7],
        tube_end_mm=[x_dim/2 - 7, y_dim, 7],
        radius_mm=3,
        consider_partial_volume=True,
        adhere_to_deformation=True
    )

    vessel_2_dictionary = sp.define_elliptical_tubular_structure_settings(
        priority=3,
        molecular_composition=sp.TISSUE_LIBRARY.blood(oxygenation=0.8),
        tube_start_mm=[x_dim/2 + 10, 0, 4],
        tube_end_mm=[x_dim/2 + 10, y_dim, 4],
        radius_mm=3,
        eccentricity=0.9,
        consider_partial_volume=True,
        adhere_to_deformation=True
    )

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    tissue_dict["muscle"] = muscle_dictionary
    tissue_dict["vessel_1"] = vessel_1_dictionary
    tissue_dict["vessel_2"] = vessel_2_dictionary
    return tissue_dict


def create_forearm_segmentation_tissue(segmentation_file_path, spacing):
    seg, header = nrrd.read(segmentation_file_path)
    input_spacing = 0.15625
    seg = np.reshape(seg[:, :, 0], (256, 1, 128))
    # plt.imshow(seg[:, 0, :])
    # plt.show()
    segmentation_volume_tiled = np.tile(seg, (1, 128, 1))
    segmentation_volume_mask = np.ones((512, 128, 320)) * 6

    segmentation_volume_mask[128:384, :, -128:] = segmentation_volume_tiled

    segmentation_volume_mask = np.round(zoom(segmentation_volume_mask, input_spacing / spacing,
                                             order=0, mode='nearest')).astype(int)
    segmentation_volume_mask[segmentation_volume_mask == 0] = 5
    segmentation_volume_mask = segmentation_volume_mask[::-1, :, :]
    return segmentation_volume_mask


def segmention_class_mapping():
    ret_dict = dict()
    ret_dict[0] = sp.TISSUE_LIBRARY.heavy_water()
    ret_dict[1] = sp.TISSUE_LIBRARY.blood(oxygenation=0.99)
    ret_dict[2] = sp.TISSUE_LIBRARY.epidermis(0.005)
    ret_dict[3] = sp.TISSUE_LIBRARY.soft_tissue(blood_volume_fraction=0.005)
    ret_dict[4] = sp.TISSUE_LIBRARY.mediprene()
    ret_dict[5] = sp.TISSUE_LIBRARY.ultrasound_gel()
    ret_dict[6] = sp.TISSUE_LIBRARY.heavy_water()
    ret_dict[7] = sp.TISSUE_LIBRARY.muscle(blood_volume_fraction=0.2)
    ret_dict[8] = sp.TISSUE_LIBRARY.blood(oxygenation=0)
    ret_dict[9] = sp.TISSUE_LIBRARY.heavy_water()
    ret_dict[10] = sp.TISSUE_LIBRARY.heavy_water()
    return ret_dict


def create_realistic_forearm_tissue(settings):
    x_dim = settings[Tags.DIM_VOLUME_X_MM]
    y_dim = settings[Tags.DIM_VOLUME_Y_MM]

    background_dictionary = sp.Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-10, 1e-10, 1.0)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    tissue_dict["muscle"] = sp.define_horizontal_layer_structure_settings(z_start_mm=1.5, thickness_mm=100,
                                                                          molecular_composition=sp.TISSUE_LIBRARY.
                                                                          soft_tissue(blood_volume_fraction=0.05),
                                                                          priority=1,
                                                                          consider_partial_volume=True,
                                                                          adhere_to_deformation=True)
    tissue_dict["epidermis"] = sp.define_horizontal_layer_structure_settings(z_start_mm=1.5, thickness_mm=0.05,
                                                                             molecular_composition=sp.TISSUE_LIBRARY.
                                                                             epidermis(0.01),
                                                                             priority=8,
                                                                             consider_partial_volume=True,
                                                                             adhere_to_deformation=True)
    tissue_dict["main_artery"] = sp.define_circular_tubular_structure_settings(
        tube_start_mm=[x_dim/2 - 4.4, 0, 5.5],
        tube_end_mm=[x_dim/2 - 4.4, y_dim, 5.5],
        molecular_composition=sp.TISSUE_LIBRARY.blood(0.99),
        radius_mm=1.25, priority=3, consider_partial_volume=True,
        adhere_to_deformation=True
    )
    tissue_dict["accomp_vein_1"] = sp.define_elliptical_tubular_structure_settings(
        tube_start_mm=[x_dim/2 - 6.8, 0, 5.6],
        tube_end_mm=[x_dim/2 - 6.8, y_dim, 5.6],
        molecular_composition=sp.TISSUE_LIBRARY.blood(0.9),
        radius_mm=0.6, priority=3, consider_partial_volume=True,
        adhere_to_deformation=True,
        eccentricity=0.8,
    )
    tissue_dict["accomp_vein_2"] = sp.define_elliptical_tubular_structure_settings(
        tube_start_mm=[x_dim / 2 - 1.25, 0, 5.6],
        tube_end_mm=[x_dim / 2 - 1.25, y_dim, 5.6],
        molecular_composition=sp.TISSUE_LIBRARY.blood(0.6),
        radius_mm=0.65, priority=3, consider_partial_volume=True,
        adhere_to_deformation=True,
        eccentricity=0.9,
    )

    tissue_dict["vessel_3"] = sp.define_elliptical_tubular_structure_settings(
        tube_start_mm=[x_dim - 6.125, 0, 3.5],
        tube_end_mm=[x_dim - 6.125, y_dim, 3.5],
        molecular_composition=sp.TISSUE_LIBRARY.blood(0.99),
        radius_mm=0.65, priority=3, consider_partial_volume=True,
        adhere_to_deformation=False,
        eccentricity=0.93,
    )

    tissue_dict["vessel_4"] = sp.define_elliptical_tubular_structure_settings(
        tube_start_mm=[5.2, 0, 4.5],
        tube_end_mm=[5.2, y_dim, 4.5],
        molecular_composition=sp.TISSUE_LIBRARY.blood(0.5),
        radius_mm=0.1, priority=3, consider_partial_volume=True,
        adhere_to_deformation=False,
        eccentricity=0,
    )

    tissue_dict["vessel_5"] = sp.define_elliptical_tubular_structure_settings(
        tube_start_mm=[x_dim - 10.4, 0, 6],
        tube_end_mm=[x_dim - 10.4, y_dim, 6],
        molecular_composition=sp.TISSUE_LIBRARY.blood(0.5),
        radius_mm=0.1, priority=3, consider_partial_volume=True,
        adhere_to_deformation=False,
        eccentricity=0,
    )

    tissue_dict["vessel_6"] = sp.define_elliptical_tubular_structure_settings(
        tube_start_mm=[x_dim - 14.4, 0, 3],
        tube_end_mm=[x_dim - 14.4, y_dim, 3],
        molecular_composition=sp.TISSUE_LIBRARY.soft_tissue(blood_volume_fraction=0.2),
        radius_mm=0.1, priority=3, consider_partial_volume=True,
        adhere_to_deformation=False,
        eccentricity=0.99,
    )

    return tissue_dict


def create_forearm_dataset_tissue(settings):
    x_dim = settings[Tags.DIM_VOLUME_X_MM]
    y_dim = settings[Tags.DIM_VOLUME_Y_MM]
    z_dim = settings[Tags.DIM_VOLUME_Z_MM]

    background_dictionary = sp.Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-10, 1e-10, 1.0)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    tissue_dict["muscle"] = sp.define_horizontal_layer_structure_settings(z_start_mm=1.5, thickness_mm=100,
                                                                          molecular_composition=sp.TISSUE_LIBRARY.
                                                                          soft_tissue(blood_volume_fraction=0.05),
                                                                          priority=1,
                                                                          consider_partial_volume=True,
                                                                          adhere_to_deformation=True)
    tissue_dict["epidermis"] = sp.define_horizontal_layer_structure_settings(z_start_mm=1.5, thickness_mm=0.05,
                                                                             molecular_composition=sp.TISSUE_LIBRARY.
                                                                             epidermis(0.01),
                                                                             priority=8,
                                                                             consider_partial_volume=True,
                                                                             adhere_to_deformation=True)

    max_number_superficial_veins = 4
    max_number_deep_veins = 4
    max_number_arteries = 4

    number_superficial_veins = np.random.randint(0, max_number_superficial_veins)
    number_deep_veins = np.random.randint(0, max_number_deep_veins)
    number_arteries = np.random.randint(1, max_number_arteries)

    for superficial_vein in range(number_superficial_veins):
        depth = sp.utils.calculate.positive_gauss(4 + sp.MorphologicalTissueProperties.SUBCUTANEOUS_VEIN_DEPTH_MEAN_MM,
                                                  sp.MorphologicalTissueProperties.SUBCUTANEOUS_VEIN_DEPTH_STD_MM)
        radius = sp.utils.calculate.positive_gauss(
            sp.MorphologicalTissueProperties.SUBCUTANEOUS_VEIN_DIAMETER_MEAN_MM/2,
            sp.MorphologicalTissueProperties.SUBCUTANEOUS_VEIN_DIAMETER_MEAN_MM/2)
        oxygenation = sp.utils.calculate.positive_gauss(sp.OpticalTissueProperties.VENOUS_OXYGENATION,
                                                        sp.OpticalTissueProperties.VENOUS_OXYGENATION_VARIATION)
        x_position = np.random.uniform(5, x_dim-5)
        eccentricity = np.random.uniform(0.2, 0.9)

        tissue_dict[f"superficial_vein_{superficial_vein}"] = sp.define_elliptical_tubular_structure_settings(
            tube_start_mm=[x_position, 0, depth],
            tube_end_mm=[x_position, y_dim, depth],
            molecular_composition=sp.TISSUE_LIBRARY.blood(oxygenation),
            radius_mm=radius, priority=3, consider_partial_volume=True,
            adhere_to_deformation=True,
            eccentricity=eccentricity,
        )

    for deep_vein in range(number_deep_veins):
        depth = np.random.uniform(4 + sp.MorphologicalTissueProperties.SUBCUTANEOUS_VEIN_DEPTH_MEAN_MM +
                                  sp.MorphologicalTissueProperties.SUBCUTANEOUS_VEIN_DEPTH_STD_MM,
                                  0.7 * z_dim)
        radius = sp.utils.calculate.positive_gauss(sp.MorphologicalTissueProperties.MEDIAN_VEIN_DIAMETER_MEAN_MM / 2,
                                                   sp.MorphologicalTissueProperties.MEDIAN_VEIN_DIAMETER_STD_MM / 2)
        oxygenation = sp.utils.calculate.positive_gauss(sp.OpticalTissueProperties.VENOUS_OXYGENATION,
                                                        sp.OpticalTissueProperties.VENOUS_OXYGENATION_VARIATION)
        x_position = np.random.uniform(5, x_dim - 5)
        eccentricity = np.random.uniform(0.2, 0.9)

        tissue_dict[f"deep_vein_{deep_vein}"] = sp.define_elliptical_tubular_structure_settings(
            tube_start_mm=[x_position, 0, depth],
            tube_end_mm=[x_position, y_dim, depth],
            molecular_composition=sp.TISSUE_LIBRARY.blood(oxygenation),
            radius_mm=radius, priority=3, consider_partial_volume=True,
            adhere_to_deformation=True,
            eccentricity=eccentricity,
        )

    for artery in range(number_arteries):
        depth = np.random.uniform(4 + 0.1 * z_dim,
                                  0.6 * z_dim)
        if np.random.uniform(0, 1) > 0.5:
            radius = sp.utils.calculate.positive_gauss(
                sp.MorphologicalTissueProperties.RADIAL_ARTERY_DIAMETER_MEAN_MM / 2,
                sp.MorphologicalTissueProperties.RADIAL_ARTERY_DIAMETER_STD_MM / 2)
        else:
            radius = sp.utils.calculate.positive_gauss(
                sp.MorphologicalTissueProperties.MEDIAN_ARTERY_DIAMETER_MEAN_MM / 2,
                sp.MorphologicalTissueProperties.MEDIAN_ARTERY_DIAMETER_STD_MM / 2)

        oxygenation = sp.utils.calculate.positive_gauss(sp.OpticalTissueProperties.ARTERIAL_OXYGENATION,
                                                        sp.OpticalTissueProperties.ARTERIAL_OXYGENATION_VARIATION)
        x_position = np.random.uniform(5, x_dim - 5)
        eccentricity = np.random.uniform(0.2, 0.8)

        tissue_dict[f"artery_{artery}"] = sp.define_elliptical_tubular_structure_settings(
            tube_start_mm=[x_position, 0, depth],
            tube_end_mm=[x_position, y_dim, depth],
            molecular_composition=sp.TISSUE_LIBRARY.blood(oxygenation),
            radius_mm=radius, priority=3, consider_partial_volume=True,
            adhere_to_deformation=True,
            eccentricity=eccentricity,
        )

    return tissue_dict
