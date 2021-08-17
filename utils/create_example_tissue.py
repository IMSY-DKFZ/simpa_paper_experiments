from simpa.utils import Tags, Settings, TISSUE_LIBRARY, MolecularCompositionGenerator, Molecule
from simpa.utils.libraries.spectra_library import AnisotropySpectrumLibrary
from simpa.utils.libraries.structure_library import *


def create_square_phantom(settings):
    x_dim = settings[Tags.DIM_VOLUME_X_MM]
    y_dim = settings[Tags.DIM_VOLUME_Y_MM]
    z_dim = settings[Tags.DIM_VOLUME_Z_MM]

    background_dictionary = Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(1e-10, 1e-10, 1.0)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    water_dictionary = define_horizontal_layer_structure_settings(
        priority=2,
        molecular_composition=TISSUE_LIBRARY.heavy_water(),
        z_start_mm=0,
        thickness_mm=z_dim,
        consider_partial_volume=True,
        adhere_to_deformation=False
    )

    epidermis_dictionary = define_rectangular_cuboid_structure_settings(
        priority=3,
        molecular_composition=TISSUE_LIBRARY.epidermis(),
        start_mm=[28, 0, 28],
        extent_mm=[34, 20, 34],
        consider_partial_volume=True,
        adhere_to_deformation=False
    )

    muscle_dictionary = define_rectangular_cuboid_structure_settings(
        priority=5,
        molecular_composition=TISSUE_LIBRARY.constant(0.05, 100, 0.9),
        start_mm=[28.1, 0, 28.1],
        extent_mm=[33.8, 19.8, 33.8],
        consider_partial_volume=True,
        adhere_to_deformation=False
    )

    vessel_1_dictionary = define_circular_tubular_structure_settings(
        priority=8,
        molecular_composition=TISSUE_LIBRARY.blood(),
        tube_start_mm=[x_dim/2 + 10, 0, z_dim/2 + 10],
        tube_end_mm=[x_dim/2 + 10, y_dim, z_dim/2 + 10],
        radius_mm=3,
        consider_partial_volume=True,
        adhere_to_deformation=False
    )

    vessel_2_dictionary = define_elliptical_tubular_structure_settings(
        priority=8,
        molecular_composition=TISSUE_LIBRARY.blood(),
        tube_start_mm=[x_dim/2 - 10, 0, z_dim/2 - 10],
        tube_end_mm=[x_dim/2 - 10, y_dim, z_dim/2 - 10],
        radius_mm=3,
        eccentricity=0.8,
        consider_partial_volume=True,
        adhere_to_deformation=False
    )

    inclusion_1_dictionary = define_rectangular_cuboid_structure_settings(
        priority=8,
        molecular_composition=TISSUE_LIBRARY.blood(),
        start_mm=[30, 0, 50],
        extent_mm=[50, 20, 8],
        consider_partial_volume=False,
        adhere_to_deformation=False
    )

    inclusion_2_dictionary = define_parallelepiped_structure_settings(start_mm=[50, 0, 30],
                                                                      edge_a_mm=[-8, 0, 5],
                                                                      edge_b_mm=[0, 20, 0],
                                                                      edge_c_mm=[2, 0, 6],
                                                                      molecular_composition=TISSUE_LIBRARY.blood())

    tissue_dict = Settings()
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

    background_dictionary = Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(1e-10, 1e-10, 1.0)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND
    background_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = False

    scattering = 100
    assumed_anisotropy = 0.8

    water_dictionary = define_horizontal_layer_structure_settings(
        molecular_composition=TISSUE_LIBRARY.constant(mua=0.1, mus=scattering, g=assumed_anisotropy),
        priority=1,
        z_start_mm=0,
        thickness_mm=z_dim,
        consider_partial_volume=False
    )

    vessel_1_dictionary = define_circular_tubular_structure_settings(
        priority=5,
        tube_start_mm=[x_dim/2, 0, 1.5],
        tube_end_mm=[x_dim/2, y_dim, 1.5],
        radius_mm=0.5,
        molecular_composition=TISSUE_LIBRARY.constant(mua=3, mus=scattering, g=assumed_anisotropy),
        consider_partial_volume=False
    )

    vessel_2_dictionary = define_circular_tubular_structure_settings(
        priority=5,
        tube_start_mm=[x_dim-3, 0, 4.75],
        tube_end_mm=[x_dim-3, y_dim, 4.75],
        radius_mm=1,
        molecular_composition=TISSUE_LIBRARY.constant(mua=1, mus=scattering, g=assumed_anisotropy),
        consider_partial_volume=False
    )

    inclusion_dictionary = define_rectangular_cuboid_structure_settings(
        priority=5,
        start_mm=[3, 0, 6.5],
        extent_mm=[2, y_dim, 3],
        molecular_composition=TISSUE_LIBRARY.constant(mua=2, mus=scattering, g=assumed_anisotropy),
        consider_partial_volume=False
    )

    tissue_dict = Settings()
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

    background_dictionary = Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(1e-4, 1e-4, 0.9)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    muscle_dictionary = define_horizontal_layer_structure_settings(
        priority=1,
        molecular_composition=TISSUE_LIBRARY.constant(0.5, 10, 0.9),
        z_start_mm=0,
        thickness_mm=z_dim,
        consider_partial_volume=True,
        adhere_to_deformation=True
    )

    vessel_1_dictionary = define_circular_tubular_structure_settings(
        priority=3,
        molecular_composition=TISSUE_LIBRARY.blood(oxygenation=1),
        tube_start_mm=[x_dim/2 - 7, 0, 7],
        tube_end_mm=[x_dim/2 - 7, y_dim, 7],
        radius_mm=3,
        consider_partial_volume=True,
        adhere_to_deformation=True
    )

    vessel_2_dictionary = define_elliptical_tubular_structure_settings(
        priority=3,
        molecular_composition=TISSUE_LIBRARY.blood(oxygenation=0.8),
        tube_start_mm=[x_dim/2 + 10, 0, 4],
        tube_end_mm=[x_dim/2 + 10, y_dim, 4],
        radius_mm=3,
        eccentricity=0.9,
        consider_partial_volume=True,
        adhere_to_deformation=True
    )

    tissue_dict = Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    tissue_dict["muscle"] = muscle_dictionary
    tissue_dict["vessel_1"] = vessel_1_dictionary
    tissue_dict["vessel_2"] = vessel_2_dictionary
    return tissue_dict
