from simpa.utils import Tags, Settings, TISSUE_LIBRARY


def create_square_phantom(settings):
    """
    This is a very simple example script of how to create a tissue definition.
    It contains a muscular background, an epidermis layer on top of the muscles
    and a blood vessel.
    """
    background_dictionary = Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(1e-10, 1e-10, 1.0)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    muscle_dictionary = Settings()
    muscle_dictionary[Tags.PRIORITY] = 1
    muscle_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
    muscle_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 100]
    muscle_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.constant(0.05, 100, 0.9)
    muscle_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    muscle_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
    muscle_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    vessel_1_dictionary = Settings()
    vessel_1_dictionary[Tags.PRIORITY] = 3
    vessel_1_dictionary[Tags.STRUCTURE_START_MM] = [settings[Tags.DIM_VOLUME_X_MM]/2,
                                                    0, 10]
    vessel_1_dictionary[Tags.STRUCTURE_END_MM] = [settings[Tags.DIM_VOLUME_X_MM]/2, settings[Tags.DIM_VOLUME_Y_MM], 10]
    vessel_1_dictionary[Tags.STRUCTURE_RADIUS_MM] = 3
    vessel_1_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
    vessel_1_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    vessel_1_dictionary[Tags.ADHERE_TO_DEFORMATION] = False
    vessel_1_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    vessel_2_dictionary = Settings()
    vessel_2_dictionary[Tags.PRIORITY] = 3
    vessel_2_dictionary[Tags.STRUCTURE_START_MM] = [settings[Tags.DIM_VOLUME_X_MM]/2 -10,
                                                    0, 5]
    vessel_2_dictionary[Tags.STRUCTURE_END_MM] = [settings[Tags.DIM_VOLUME_X_MM]/2 -10, settings[Tags.DIM_VOLUME_Y_MM], 5]
    vessel_2_dictionary[Tags.STRUCTURE_RADIUS_MM] = 2
    vessel_2_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.blood()
    vessel_2_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    vessel_2_dictionary[Tags.ADHERE_TO_DEFORMATION] = False
    vessel_2_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    epidermis_dictionary = Settings()
    epidermis_dictionary[Tags.PRIORITY] = 8
    epidermis_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 1]
    epidermis_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 1.1]
    epidermis_dictionary[Tags.MOLECULE_COMPOSITION] = TISSUE_LIBRARY.epidermis()
    epidermis_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    epidermis_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
    epidermis_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    tissue_dict = Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    tissue_dict["muscle"] = muscle_dictionary
    tissue_dict["epidermis"] = epidermis_dictionary
    tissue_dict["vessel_1"] = vessel_1_dictionary
    tissue_dict["vessel_2"] = vessel_2_dictionary
    return tissue_dict
