import numpy as np
import yaml
import matplotlib.pyplot as plt

""" SW model of the design parameters of a resonant driver circuit for an induction heater

"""


def save_flattened_dict_as_ltspice_txt_file(config, file_name):
    """
    Saves a text file with a flattened version of the config dictionary to be imported
    by ltspice. NOTE: currently only the lowest level names are used as the parameter
    name so care must be taken that there are no duplicates
    Args:
        config (dict): dictionary of parameters to save
        file_name (str): name of the file to be saved
    Returns:
        None
    """
    # ltspice spice format is .param <name> = <value>
    # the name only is the key of the lowest hiearchy level
    # maximum depth supported is 3
    param_file = open(file_name + ".txt", "w")
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    for sub_sub_key, sub_sub_value in sub_value.items():
                        if not isinstance(sub_sub_value, str):
                            param_file.write(
                                f".param {sub_sub_key} = {sub_sub_value}\n"
                            )
                elif not isinstance(sub_value, str):
                    param_file.write(f".param {sub_key} = {sub_value}\n")
        # make sure valus is a number not a string
        elif not isinstance(value, str):
            param_file.write(f".param {key} = {value}\n")
    # save the file
    param_file.close()


class ResonantDriver:
    def __init__(self, ee_config):
        """
        Args:
            ee_config (dict): dictionary with the general design parameters
        """
        self.ee_config = ee_config
        return None

    def analayze(self):
        """Calculates some useful derivative parameters for the resonant driver for simulation"""
        # calculate the primary and secondary inductances

        return None

    def save_param_text_file_ltspice(self, file_name="driver_params"):
        """
        creates a text file with a flattened version of the config dictionary
        Args:
            file_name (str): name of the file to be saved
        Returns:
            None
        """
        save_flattened_dict_as_ltspice_txt_file(self.ee_config, file_name)
