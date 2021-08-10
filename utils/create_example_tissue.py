from simpa.utils import Tags, Settings, TISSUE_LIBRARY, MolecularCompositionGenerator, Molecule
from simpa.utils.libraries.spectra_library import AnisotropySpectrumLibrary
from simpa.utils.libraries.structure_library import define_parallelepiped_structure_settings


def create_square_phantom(settings):
    """
    This is a very simple example script of how to create a tissue definition.
    It contains a muscular background, an epidermis layer on top of the muscles
    and a blood vessel.
    """
    background_dictionary = Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(1e-10, 1e-10, 1.0)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    water_dictionary = Settings()
    water_dictionary[Tags.PRIORITY] = 2
    water_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
    water_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 100]
    water_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.heavy_water()
    water_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    water_dictionary[Tags.ADHERE_TO_DEFORMATION] = False
    water_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    epidermis_dictionary = Settings()
    epidermis_dictionary[Tags.PRIORITY] = 3
    epidermis_dictionary[Tags.STRUCTURE_START_MM] = [28, 0, 28]#[28, 0, 29.5]
    epidermis_dictionary[Tags.STRUCTURE_X_EXTENT_MM] = 34#33.2
    epidermis_dictionary[Tags.STRUCTURE_Y_EXTENT_MM] = 20#19.8
    epidermis_dictionary[Tags.STRUCTURE_Z_EXTENT_MM] = 34#33.2
    epidermis_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.epidermis()
    epidermis_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    epidermis_dictionary[Tags.ADHERE_TO_DEFORMATION] = False
    epidermis_dictionary[Tags.STRUCTURE_TYPE] = Tags.RECTANGULAR_CUBOID_STRUCTURE

    muscle_dictionary = Settings()
    muscle_dictionary[Tags.PRIORITY] = 5
    muscle_dictionary[Tags.STRUCTURE_START_MM] = [28.1, 0, 28.1]#[28.1, 0, 29.6]
    muscle_dictionary[Tags.STRUCTURE_X_EXTENT_MM] = 33.8#33
    muscle_dictionary[Tags.STRUCTURE_Y_EXTENT_MM] = 19.8#19
    muscle_dictionary[Tags.STRUCTURE_Z_EXTENT_MM] = 33.8#33
    muscle_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(0.05, 100, 0.9)
    muscle_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    muscle_dictionary[Tags.ADHERE_TO_DEFORMATION] = False
    muscle_dictionary[Tags.STRUCTURE_TYPE] = Tags.RECTANGULAR_CUBOID_STRUCTURE

    vessel_1_dictionary = Settings()
    vessel_1_dictionary[Tags.PRIORITY] = 8
    vessel_1_dictionary[Tags.STRUCTURE_START_MM] = [settings[Tags.DIM_VOLUME_X_MM]/2 + 10, 0, settings[Tags.DIM_VOLUME_Z_MM]/2 + 10]
    vessel_1_dictionary[Tags.STRUCTURE_END_MM] = [settings[Tags.DIM_VOLUME_X_MM]/2 + 10, settings[Tags.DIM_VOLUME_Y_MM], settings[Tags.DIM_VOLUME_Z_MM]/2 + 10]
    vessel_1_dictionary[Tags.STRUCTURE_RADIUS_MM] = 3
    vessel_1_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
    vessel_1_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    vessel_1_dictionary[Tags.ADHERE_TO_DEFORMATION] = False
    vessel_1_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    vessel_2_dictionary = Settings()
    vessel_2_dictionary[Tags.PRIORITY] = 8
    vessel_2_dictionary[Tags.STRUCTURE_START_MM] = [settings[Tags.DIM_VOLUME_X_MM]/2 - 10, 0, settings[Tags.DIM_VOLUME_Z_MM]/2 - 10]
    vessel_2_dictionary[Tags.STRUCTURE_END_MM] = [settings[Tags.DIM_VOLUME_X_MM]/2 - 10, settings[Tags.DIM_VOLUME_Y_MM], settings[Tags.DIM_VOLUME_Z_MM]/2 - 10]
    vessel_2_dictionary[Tags.STRUCTURE_RADIUS_MM] = 2
    vessel_2_dictionary[Tags.STRUCTURE_ECCENTRICITY] = 0.8
    vessel_2_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
    vessel_2_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = False
    vessel_2_dictionary[Tags.STRUCTURE_TYPE] = Tags.ELLIPTICAL_TUBULAR_STRUCTURE

    inclusion_1_dictionary = Settings()
    inclusion_1_dictionary[Tags.PRIORITY] = 8
    inclusion_1_dictionary[Tags.STRUCTURE_START_MM] = [30, 0, 50]
    inclusion_1_dictionary[Tags.STRUCTURE_X_EXTENT_MM] = 5
    inclusion_1_dictionary[Tags.STRUCTURE_Y_EXTENT_MM] = 20
    inclusion_1_dictionary[Tags.STRUCTURE_Z_EXTENT_MM] = 8
    inclusion_1_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
    inclusion_1_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = False
    inclusion_1_dictionary[Tags.STRUCTURE_TYPE] = Tags.RECTANGULAR_CUBOID_STRUCTURE

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
