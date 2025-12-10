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

    def phase_setpoint(self, R, L, C, phase_des=0):
        """Calculates the required frequency to achieve a desired phase angle
        Args:
            R (float): total resistance in the resonant tank [Ohm]
            L (float): inductance in the resonant tank [H]
            C (float): capacitance in the resonant tank [F]
            phase_des (float): desired phase angle between voltage and current [rad]
        Returns:
            f (float): required frequency to achieve the desired phase angle [Hz]
        """
        # Analytical expression for the complex impedance of the RLC circuit
        # tan(phi) = (wL - 1/(wC)) / R
        # Solve for w: wL - 1/(wC) = R * tan(phi)
        # => w^2 L - R tan(phi) w - 1/C = 0
        # Quadratic in w: a w^2 + b w + c = 0
        a = L
        b = -R * np.tan(phase_des)
        c = -1 / C

        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            raise ValueError("No real solution for the given parameters and phase_des.")

        w1 = (-b + np.sqrt(discriminant)) / (2 * a)
        w2 = (-b - np.sqrt(discriminant)) / (2 * a)
        # Choose the positive, physically meaningful solution
        w_des = w1 if w1 > 0 else w2
        f = w_des / (2 * np.pi)
        return f

    def analayze(self):
        """Calculates some useful derivative parameters for the resonant driver for simulation"""
        # calculate the primary and secondary inductances
        # indictance factor
        Np = self.ee_config["XFMR"]["N_pri"]
        Ns = self.ee_config["XFMR"]["N_sec"]
        Al_eff = self.ee_config["XFMR"]["Al"] * self.ee_config["XFMR"]["N_cores"]
        Lp = Al_eff * Np**2
        Ls = Al_eff * Ns**2
        self.ee_config["XFMR"]["Lp"] = Lp
        self.ee_config["XFMR"]["Ls"] = Ls

        R_sec_tot = (
            self.ee_config["R_load"]
            + self.ee_config["XFMR"]["ESR_sec"]
            + self.ee_config["Cr_ESR"]
        )
        # calculate the resonant frequency

        f_des = self.phase_setpoint(
            R_sec_tot,
            self.ee_config["L_coil"],
            self.ee_config["Cr"],
            phase_des=-5 * np.pi / 180,  # 5 degrees phase lead,
        )

        self.ee_config["F_res_calc"] = f_des

        print(self.ee_config)

        return None

    def save_param_text_file_ltspice(self, file_name="spice/driver_params"):
        """
        creates a text file with a flattened version of the config dictionary
        Args:
            file_name (str): name of the file to be saved
        Returns:
            None
        """
        save_flattened_dict_as_ltspice_txt_file(self.ee_config, file_name)
