from abc import ABC, abstractmethod
import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
from pymoo.operators.sampling.lhs import LHS


"""
@staticmethod: Defines a method that doesn’t depend on either the class (cls) or instance (self). Essentially a function inside a class
@property: Allows you to define a method that can be accessed like an attribute
@abstractmethod: A decorator indicating abstract methods. Abstract methods are methods that are defined in the base class but may not have an implementation. They must be implemented in the derived class
@value.setter: A decorator that allows you to define a method that can be used to set the value of a property
"""

_oneport_registry = {}


def register_oneport(cls):
    _oneport_registry[cls.__name__] = cls
    return cls


class OneportNetworkFittingProblem(Problem):
    def __init__(self, cost_f, bounds, f, Z, alpha):
        self.f = f
        self.Z = Z
        self.alpha = alpha
        self.cost_f = cost_f
        super().__init__(
            n_var=len(bounds),
            n_obj=1,
            n_constr=0,
            xl=np.array([b[0] for b in bounds]),
            xu=np.array([b[1] for b in bounds]),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # x is a population (2D array), must apply cost_f to each row
        out["F"] = np.array([self.cost_f(ind, self.f, self.Z, self.alpha) for ind in x])


# time domain version of the fitting problem
# variable naming based on thermal systems
class OneportNetworkTDFittingProblem(Problem):
    def __init__(self, cost_f, bounds, t, u, y):
        self.t = t
        self.u = u
        self.y = y
        self.cost_f = cost_f
        super().__init__(
            n_var=len(bounds),
            n_obj=1,
            n_constr=0,
            xl=np.array([b[0] for b in bounds]),
            xu=np.array([b[1] for b in bounds]),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # x is a population (2D array), must apply cost_f to each row
        out["F"] = np.array([self.cost_f(ind, self.t, self.u, self.y) for ind in x])
        # print(out)


class AnnealingCallback(Callback):
    def __init__(self, algorithm, max_gen, start_F=1.2, end_F=0.5):
        super().__init__()
        self.algorithm = algorithm
        self.max_gen = max_gen
        self.start_F = start_F
        self.end_F = end_F

    def notify(self, algorithm):
        gen = algorithm.n_gen
        t = gen / self.max_gen
        # Linear annealing schedule
        algorithm.F = (1 - t) * self.start_F + t * self.end_F


class OnePortNetwork(ABC):
    def __init__(self, N):
        self.N = N
        # self.param_dict = {}  # gets define in child oneport classes
        # Impedance Form (Z(s))
        self.Az = np.zeros((self.N, self.N))
        self.Bz = np.zeros((self.N, 1))
        self.Cz = np.zeros((1, self.N))
        self.Dz = np.zeros((1, 1))
        self.kappa_Z = None  # impedance system condition number
        # Admittance Form (Y(s))
        self.Ay = np.zeros((self.N, self.N))
        self.By = np.zeros((self.N, 1))
        self.Cy = np.zeros((1, self.N))
        # output for the current of the first mode only
        self.Cy1 = np.zeros((1, self.N))
        self.Cy1[0, 0] = 1  # output for the current of the first mode only
        self.Dy = np.zeros((1, 1))
        self.kappa_Y = None  # admittance system condition number
        # Admittance Derivative Form (s*Y(s))
        self.Ays = np.zeros((self.N, self.N))
        self.Bys = np.zeros((self.N, 1))
        self.Cys = np.zeros((1, self.N))
        self.Dys = np.zeros((1, 1))
        self.kappa_Ys = None  # admittance derivative system condition number
        # Impedance Integral Form (Z(s)/s)
        self.Az_s = np.zeros((self.N, self.N))
        self.Bz_s = np.zeros((self.N, 1))
        self.Cz_s = np.zeros((1, self.N))
        self.Dz_s = np.zeros((1, 1))
        self.kappa_Z_s = None  # impedance integral system condition number

        # intialize list of poles and zeros (when in impedance form)
        # the combination of poles, zeros, and gain encode the same information as any of the state space variants
        self.poles = []
        self.zeros = []
        self.Z0 = None  # DC impedance

        # relation to power dissipation
        # vector of resistor currents is: i_R = Tx * i_L + Tu * u  where units of u depend on the network type
        # Cauer RL and RLR networks have u = vin, Cauer LR and LRL networks have u = iin by default
        # depending on the network type Tu may be zero, a constant matrix, or a function of the lumped element parameters. Tx is always a constant matrix
        self.Tx = None
        self.Tu = None

        # discrete time state space matrices that are populated only for the requested state space variant (impedance/ admittance) and time step
        self.dt = None  # time step for discrete time system
        self.A_dt = np.zeros((self.N, self.N))
        self.B_dt = np.zeros((self.N, 1))
        self.C_dt = np.zeros((1, self.N))
        self.D_dt = np.zeros((1, 1))

    def format_matrix(self, M, sig_digits=1):
        """Formats a matrix into aligned scientific notation or integers if sig_digits is 0."""
        if M.size == 0:
            return "[]"

        if sig_digits == 0:
            # Convert all elements to integers
            formatted = np.vectorize(lambda x: f"{int(x)}")(M)
        else:
            # Convert all elements to scientific notation with fixed significant digits
            formatted = np.vectorize(lambda x: f"{x:.{sig_digits}e}")(M)

        # Determine column widths (max length in each column)
        col_widths = [max(len(row[i]) for row in formatted) for i in range(M.shape[1])]

        # Format each row with proper spacing
        rows = [
            "  ".join(f"{val:>{col_widths[i]}}" for i, val in enumerate(row))
            for row in formatted
        ]

        return "[" + "\n ".join(rows) + "]"

    def __str__(self):
        print_str = "\t Order: {}\n".format(self.N)
        # if parameters are initialized print them
        param_prefixs = {
            "R": "Resistance",
            "L": "Inductance",
            "C": "Capacitance",
        }  # add more as needed
        for key, val in self.param_dict.items():
            if key in param_prefixs.keys() and val is not None:
                print_str += "\t {}: {}\n".format(param_prefixs[key], val)
        # if poles and zeros are calculated print them
        if self.zeros is not []:
            print_str += "\t Z(s) Zeros: {}\n".format(self.zeros)
        if self.poles is not []:
            print_str += "\t Z(s) Poles: {}\n".format(self.poles)
        if self.Z0 is not None:
            print_str += "\t Z(0): {}\n".format(self.Z0)
        # add Impedance state space matrices
        if np.any(self.Az):
            print_str += "Impedance State Space Matrices:\n"
            print_str += f"A = \n{self.format_matrix(self.Az)}\n\n"
            print_str += f"B = \n{self.format_matrix(self.Bz)}\n\n"
            print_str += f"C = \n{self.format_matrix(self.Cz)}\n\n"
            print_str += f"D = \n{self.format_matrix(self.Dz)}\n\n"
        if np.any(self.Ay):
            print_str += "Admittance State Space Matrices:\n"
            print_str += f"A = \n{self.format_matrix(self.Ay)}\n\n"
            print_str += f"B = \n{self.format_matrix(self.By)}\n\n"
            print_str += f"C = \n{self.format_matrix(self.Cy)}\n\n"
            print_str += f"D = \n{self.format_matrix(self.Dy)}\n\n"
        if np.any(self.Az_s):
            print_str += "Impedance Integral State Space Matrices:\n"
            print_str += f"A = \n{self.format_matrix(self.Az_s)}\n\n"
            print_str += f"B = \n{self.format_matrix(self.Bz_s)}\n\n"
            print_str += f"C = \n{self.format_matrix(self.Cz_s)}\n\n"
            print_str += f"D = \n{self.format_matrix(self.Dz_s)}\n\n"
        if np.any(self.Ays):
            print_str += "Admittance Derivative State Space Matrices:\n"
            print_str += f"A = \n{self.format_matrix(self.Ays)}\n\n"
            print_str += f"B = \n{self.format_matrix(self.Bys)}\n\n"
            print_str += f"C = \n{self.format_matrix(self.Cys)}\n\n"
            print_str += f"D = \n{self.format_matrix(self.Dys)}\n\n"
        # add Tx and Tu matrices
        if self.Tx is not None:
            print_str += "Tx = \n{}\n\n".format(
                self.format_matrix(self.Tx, sig_digits=0)
            )
        if self.Tu is not None:
            print_str += "Tu = \n{}\n\n".format(
                self.format_matrix(self.Tu, sig_digits=2)
            )
        return print_str

    def plot_pz(self):
        """makes multiple plots to depict the system dynamics"""
        Y_sys = None
        Z_sys = None
        if np.any(self.Ay):
            Y_sys = ctrl.ss2tf(self.Ay, self.By, self.Cy, self.Dy)
        if np.any(self.Az):
            Z_sys = ctrl.ss2tf(self.Az, self.Bz, self.Cz, self.Dz)

        # make all the valid plots (if D!=0 cannot make an impulse response plot, if DC gain is 0 cannot make a step response plot)
        systems = {"Impedance": Z_sys, "Admittance": Y_sys}
        # make sa 2x2 figure of step and impulse response for both impedance and admittance forms
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        for i, (name, sys) in enumerate(systems.items()):
            if sys is None:
                # make both plots blank
                axs[0, i].set_title(f"{name} Step Response (Not Defined)")
                axs[1, i].set_title(f"{name} Impulse Response (Not Defined)")
            else:
                dc_gain = ctrl.dcgain(sys)
                sys_ss = ctrl.tf2ss(sys)
                # plot if the DC gain is finite (not close to 0 or infinity)
                if dc_gain < 1e-6 or dc_gain > 1e8:
                    axs[0, i].set_title(f"{name} Step Response (Not Defined)")
                else:
                    t_step, y_step = ctrl.step_response(sys)
                    axs[0, i].plot(t_step, y_step)
                    axs[0, i].set_title(f"{name} Step Response")
                    axs[0, i].set_xlabel("Time [s]")
                    axs[0, i].set_ylabel("Current [A]")
                    axs[0, i].grid(True)
                # Plot the impulse respone only if D==0
                if sys_ss.D != 0:
                    axs[1, i].set_title(f"{name} Impulse Response (Not Defined)")
                else:
                    t_imp, y_imp = ctrl.impulse_response(sys)
                    axs[1, i].plot(t_imp, y_imp)
                    axs[1, i].set_title(f"{name} Impulse Response")
                    axs[1, i].set_xlabel("Time [s]")
                    axs[1, i].set_ylabel("Current [A]")
                    axs[1, i].grid(True)

    @abstractmethod
    def pack_params(self):
        pass

    @abstractmethod
    def unpack_params(self, params_arr):
        pass

    @abstractmethod
    def estimate_bounds(self, f, Z):
        pass

    @abstractmethod
    def eval_Z(self, f):
        pass

    def eval_P_inst(self, state, u=None):
        """evaluates the instantaneous power dissipated in the electrical network
        Args:
            state (array): [self.N] array of state variables (inductor currents)
            u (float): system input (either voltage or current depending on the network type)
        Returns:
            P (float): instantaneous power dissipated in the network across all resistive elements
        """
        # first calculate i_r
        state = np.reshape(state, (self.N, 1))
        if u is None:
            u = 0
        i_R = self.Tx @ state + self.Tu * u  # [N_r, 1] array of resistor currents
        R_mat = np.diag(self.param_dict["R"])
        P_tot = i_R.T @ R_mat @ i_R
        return P_tot[0, 0]

    def cost_f(self, params_arr, f, Z, alpha=1):
        """frequency domain cost function biases low frequency normalized impedance error"""
        self.unpack_params(params_arr)
        Z_fit = self.eval_Z(f)
        w = lf_weighting(f, alpha=alpha)
        cost = np.sum(np.square(np.abs(Z_fit - Z) / np.abs(Z)) * w) / len(f)
        return cost

    def cost_f_td(self, params_arr, t, u, y):
        """time domain cost is simply the mean square error between the y_meas and y_fit"""
        self.unpack_params(params_arr)
        self.make_state_space()
        y_fit = self.eval_td(t, u)

        cost = np.mean(np.square(np.abs(y_fit - y)))
        return cost

    def fit_network(self, f, Z, alpha=3, opt_kwargs=None):
        """estimate the parameters of the network from impedance data
        Args:
            f (array): Frequency array
            Z (complex array): Impedance array
            alpha (float): Weighting factor for low frequency data see: lf_weighting()
            opt_kwargs (dict): Optional arguments for the optimization method
        """
        bounds = self.estimate_bounds(f, Z)
        problem = OneportNetworkFittingProblem(self.cost_f, bounds, f, Z, alpha)

        # unpack the optimization kwargs
        if opt_kwargs is None:
            opt_kwargs = {}
        pop_size = opt_kwargs.get("pop_size", 100)
        max_gen = opt_kwargs.get("max_gen", 100)
        start_F = opt_kwargs.get("start_F", 0.5)
        end_F = opt_kwargs.get("end_F", 0.05)
        CR = opt_kwargs.get("CR", 0.5)

        # Setup DE algorithm with annealing-style parameters
        de = DE(
            pop_size=pop_size,
            sampling=LHS(),  # Latin Hypercube Sampling
            variant="DE/rand/1/bin",  # standard DE
            CR=CR,  # recombination
            F=start_F,  # differential weight (mutation factor)
            dither="vector",  # enables element-wise dither (annealing-like)
            jitter=False,
        )

        termination = get_termination("n_gen", max_gen)

        # Attach dynamic annealing behavior
        callback = AnnealingCallback(de, max_gen, start_F=start_F, end_F=end_F)

        # print each input to minimize
        res = minimize(
            problem,
            de,
            termination=termination,
            seed=42,
            callback=callback,
            verbose=False,
            save_history=False,
        )

        self.unpack_params(res.X)

    def fit_network_td(self, t, u, y, opt_kwargs={}):
        """estimate system parameters based on time domain masured data
        Args:
            t (array): Time array [Nt]
            u (array): Input array [Nt]
            y (array): Output array [Nt]
            opt_kwargs (dict): Optional arguments for the optimization method
        """
        # estimate both the consntant initial condition and the system parameters
        bounds = self.estimate_bounds(t, u, y)
        problem = OneportNetworkTDFittingProblem(self.cost_f_td, bounds, t, u, y)

        # unpack the optimization kwargs
        if opt_kwargs is None:
            opt_kwargs = {}
        pop_size = opt_kwargs.get("pop_size", 100)
        max_gen = opt_kwargs.get("max_gen", 100)
        start_F = opt_kwargs.get("start_F", 0.5)
        end_F = opt_kwargs.get("end_F", 0.05)
        CR = opt_kwargs.get("CR", 0.5)

        # Setup DE algorithm with annealing-style parameters
        de = DE(
            pop_size=pop_size,
            sampling=LHS(),  # Latin Hypercube Sampling
            variant="DE/rand/1/bin",  # standard DE
            CR=CR,  # recombination
            F=start_F,  # differential weight (mutation factor)
            dither="vector",  # enables element-wise dither (annealing-like)
            jitter=False,
        )

        termination = get_termination("n_gen", max_gen)

        # Attach dynamic annealing behavior
        callback = AnnealingCallback(de, max_gen, start_F=start_F, end_F=end_F)

        # print each input to minimize
        res = minimize(
            problem,
            de,
            termination=termination,
            seed=42,
            callback=callback,
            verbose=False,
            save_history=False,
        )

        self.unpack_params(res.X)

    @staticmethod
    def invert_state_space(A, B, C, D):
        """if the system is invertible, return an inverted system"""
        # check that D is nonzero
        if D == 0:
            raise ValueError("D matrix is zero, system is not invertible")
        D_inv = np.linalg.inv(D)
        A_inv = A - B @ D_inv @ C
        B_inv = B @ D_inv
        C_inv = -D_inv @ C
        return A_inv, B_inv, C_inv, D_inv

    @staticmethod
    def condition_number(A):
        """Calculate the condition number of a matrix"""
        return np.linalg.cond(A)

    @abstractmethod
    def make_state_space(self):
        pass

    def discretize_ss(self, dt, type="Z"):
        """Converts the specified state space representation to discrete time if it exists
        Args:
            dt (float): time step for the discrete time system
            type (str): type of state space representation to convert to discrete time
                options are currently only 'Z' or 'Y' for impedance or admittance form
        """
        if type == "Z" and np.any(self.Az):
            ct_sys = ctrl.StateSpace(self.Az, self.Bz, self.Cz, self.Dz)
            dt_sys = ctrl.c2d(ct_sys, dt)
        elif type == "Y" and np.any(self.Ay):
            ct_sys = ctrl.StateSpace(self.Ay, self.By, self.Cy, self.Dy, dt)
            dt_sys = ctrl.c2d(ct_sys, dt)
        else:
            raise ValueError("State space representation not defined or invlid type")
        self.A_dt = dt_sys.A
        self.B_dt = dt_sys.B
        self.C_dt = dt_sys.C
        self.D_dt = dt_sys.D
        self.dt = dt

    def make_flux_sys(self, A, B, C, D):
        """Converts the default system representations (which use inductor currents as state variables)
        to representations that use flux states as state variables. This is useful for better numerical
        conditioning for impedance synthesis.
        Args:
            A (array): State matrix
            B (array): Input matrix
            C (array): Output matrix
            D (array): Feedforward matrix
        """
        # get the last N inductnace values from the system
        L_arr = self.param_dict["L"]
        # pick the last self .N inductances
        L = L_arr[-self.N :]
        # populate along diagonal
        L_diag = np.diag(L)
        A_flux = L_diag @ A @ np.linalg.inv(L_diag)
        B_flux = L_diag @ B
        C_flux = C @ np.linalg.inv(L_diag)
        D_flux = D
        return A_flux, B_flux, C_flux, D_flux

    def make_balanced_sys(self, A, B, C, D, order_reduction=0):
        """Converts the default system representations (which use inductor currents as state variables)
        to representations that use balanced state variables. This is useful for better numerical
        conditioning for impedance synthesis, but does not enforce the states being physical
        """
        sys = ctrl.ss(A, B, C, D)
        sys_balanced = ctrl.balred(sys, self.N - int(order_reduction))
        return sys_balanced.A, sys_balanced.B, sys_balanced.C, sys_balanced.D

    def pz_from_ss(self):
        """get the poles and zeros as well as the DC gain from the state space representation"""
        # system has has Impedance Form (Z(s))
        if np.any(self.Az):
            sys = ctrl.ss2tf(self.Az, self.Bz, self.Cz, self.Dz)
            poles, zeros = ctrl.pole_zero_map(sys)
            DC_gain = ctrl.dcgain(sys)
            self.Z0 = DC_gain
            self.poles = poles
            self.zeros = zeros
        # system has Admittance Form (Y(s))
        elif np.any(self.Ay):
            sys = ctrl.ss2tf(self.Ay, self.By, self.Cy, self.Dy)
            poles, zeros = ctrl.pole_zero_map(sys)
            DC_gain = ctrl.dcgain(sys)
            self.Z0 = 1 / DC_gain
            # poles of admittance are zeros of impedance and vice versa
            self.poles = zeros
            self.zeros = poles
        # system is in Impedance Integral Form (Z(s)/s)
        elif np.any(self.Az_s):
            sys = ctrl.ss2tf(self.Az_s, self.Bz_s, self.Cz_s, self.Dz_s)
            poles, zeros = ctrl.pole_zero_map(sys)
            # add the zero at the origin to get into impedance form
            zeros = np.append(zeros, 0)
            DC_gain = ctrl.dcgain(sys)
            self.Z0 = 0  # in this form the DC impedance is always zero
            self.poles = poles
            self.zeros = zeros
        else:
            raise ValueError("State space matrices not yet defined")

    def analyze_numerical_stability(
        self, test_flux_state=True, test_balanced_state=True
    ):
        """calculates the condition number for each populated state space matrix
        Args
            test_flux_state (bool): If True, ss systems are created with flux as state variables
        """
        # for each of Az, Ay, Az_s, Ays calculate the condition number
        if np.any(self.Az):
            kappa_Z = self.condition_number(self.Az)
            print(f"Impedance State Space Condition Number: {kappa_Z}")
            if test_flux_state:
                A_flux, B_flux, C_flux, D_flux = self.make_flux_sys(
                    self.Az, self.Bz, self.Cz, self.Dz
                )
                kappa_Z_flux = self.condition_number(A_flux)
                print(f"Impedance State Space Flux Condition Number: {kappa_Z_flux}")
            if test_balanced_state:
                A_bal, B_bal, C_bal, D_bal = self.make_balanced_sys(
                    self.Az, self.Bz, self.Cz, self.Dz
                )
                kappa_Z_bal = self.condition_number(A_bal)
                print(f"Impedance State Space Balanced Condition Number: {kappa_Z_bal}")
        if np.any(self.Ay):
            kappa_Y = self.condition_number(self.Ay)
            print(f"Admittance State Space Condition Number: {kappa_Y}")
            if test_flux_state:
                A_flux, B_flux, C_flux, D_flux = self.make_flux_sys(
                    self.Ay, self.By, self.Cy, self.Dy
                )
                kappa_Y_flux = self.condition_number(A_flux)
                print(f"Admittance State Space Flux Condition Number: {kappa_Y_flux}")
            if test_balanced_state:
                A_bal, B_bal, C_bal, D_bal = self.make_balanced_sys(
                    self.Ay, self.By, self.Cy, self.Dy
                )
                kappa_Y_bal = self.condition_number(A_bal)
                print(
                    f"Admittance State Space Balanced Condition Number: {kappa_Y_bal}"
                )
        if np.any(self.Az_s):
            kappa_Z_s = self.condition_number(self.Az_s)
            print(f"Impedance Integral State Space Condition Number: {kappa_Z_s}")
            if test_flux_state:
                A_flux, B_flux, C_flux, D_flux = self.make_flux_sys(
                    self.Az_s, self.Bz_s, self.Cz_s, self.Dz_s
                )
                kappa_Z_s_flux = self.condition_number(A_flux)
                print(
                    f"Impedance Integral State Space Flux Condition Number: {kappa_Z_s_flux}"
                )
            if test_balanced_state:
                A_bal, B_bal, C_bal, D_bal = self.make_balanced_sys(
                    self.Az_s, self.Bz_s, self.Cz_s, self.Dz_s
                )
                kappa_Z_s_bal = self.condition_number(A_bal)
                print(
                    f"Impedance Integral State Space Balanced Condition Number: {kappa_Z_s_bal}"
                )
        if np.any(self.Ays):
            kappa_Ys = self.condition_number(self.Ays)
            print(f"Admittance Derivative State Space Condition Number: {kappa_Ys}")
            if test_flux_state:
                A_flux, B_flux, C_flux, D_flux = self.make_flux_sys(
                    self.Ays, self.Bys, self.Cys, self.Dys
                )
                kappa_Ys_flux = self.condition_number(A_flux)
                print(
                    f"Admittance Derivative State Space Flux Condition Number: {kappa_Ys_flux}"
                )
            if test_balanced_state:
                A_bal, B_bal, C_bal, D_bal = self.make_balanced_sys(
                    self.Ays, self.Bys, self.Cys, self.Dys
                )
                kappa_Ys_bal = self.condition_number(A_bal)
                print(
                    f"Admittance Derivative State Space Balanced Condition Number: {kappa_Ys_bal}"
                )

    @abstractmethod
    def make_interesting_params(self, f_bnds):
        pass

    @abstractmethod
    def test_network(self, order, f_bnds):
        pass


@register_oneport
class Cauer_RL(OnePortNetwork):
    def __init__(self, N=3, params=None, T0=0, rho_alpha=0.00393):

        self.type = "Cauer_RL"
        # if params are given in a dictionary make sure keys are 'R' and 'L'
        if params and "R" in params and "L" in params:
            if len(params["R"]) != N or len(params["L"]) != N:
                raise ValueError("Length of R and L should be equal to order")
            self.param_dict = {"R": params["R"], "L": params["L"]}
            super().__init__(N)
            self.make_state_space()
        else:
            self.param_dict = {"R": np.zeros(N), "L": np.zeros(N)}
            super().__init__(N)

        # define the Tx and Tu matrices for power dissipation
        # Tx is upper triangular ones matrix of size N x N
        self.Tx = np.triu(np.ones((self.N, self.N)), 0)
        self.Tu = np.zeros((self.N, 1))  # No feedthrough

    def __str__(self):
        print_str = ""
        print_str += "Cauer RL Network\n"
        # add string from parent class
        print_str += super().__str__()
        return print_str

    def pack_params(self, param_dict):
        """make a 1D array of log10 parameters to optimize over"""
        param_arr = np.concatenate(
            (np.log10(param_dict["R"]), np.log10(param_dict["L"]))
        )
        return param_arr

    def unpack_params(self, param_arr):
        """unpack the 1D array of log10 parameters"""
        params = {}
        params["R"] = 10 ** param_arr[: self.N]
        params["L"] = 10 ** param_arr[self.N :]
        self.param_dict["R"] = params["R"]
        self.param_dict["L"] = params["L"]

    def estimate_bounds(self, f, Z):
        """estimates conservative bounds for each parameter based on the data"""
        # calculate R and L
        R = np.real(Z)
        L = np.imag(Z) / (2 * np.pi * f)
        L_dc = L[0]
        R_min = np.min(R) / 10  # 10 times smaller than the LF resistance
        R_max = 1e3 * np.max(R)  # 1000 times max real impedance
        L_min = 1e-6 * L_dc
        L_max = 1e2 * L_dc

        # assemble the bounds in the for [(min, max)] for each parameter
        bounds = self.N * [(R_min, R_max)] + self.N * [(L_min, L_max)]
        # take log10 of every value
        bounds = [(np.log10(b[0]), np.log10(b[1])) for b in bounds]
        return bounds

    def eval_Z(self, f):
        """iteratively add to the one port impedance"""
        s = 1j * 2 * np.pi * f
        R = self.param_dict["R"]
        L = self.param_dict["L"]
        # last RL pair first
        Z = R[-1] + L[-1] * s
        # add the effect of the rest of the pairs
        for i in range(self.N - 2, -1, -1):
            Z = R[i] + 1 / (1 / (L[i] * s) + 1 / Z)
        return Z

    def make_state_space(self):
        """convert RL network to state space"""
        N = self.N
        R = self.param_dict["R"]
        L = self.param_dict["L"]

        A = np.zeros((N, N))
        B = np.zeros((N, 1))
        for i in range(N):
            B[i, 0] = 1 / L[i]
            for j in range(N):
                A[i, j] = -np.sum(R[: min(i, j) + 1]) / L[i]
        C = np.ones((1, N))
        D = np.zeros((1, 1))

        self.Ay[:] = A
        self.By[:] = B
        self.Cy[:] = C
        self.Dy[:] = D
        # calculate poles and zeros of Z(s)
        self.pz_from_ss()

    def make_interesting_params(self, f_bnds=[1, 1e6]):
        """Generate interesting parameters for the RL network.
        Args:
            f_bnds (list): Frequency bounds of intetest
        Returns:
            param_dict: (dict) Dictionary of parameters that yield an interesting response
        """
        decade_buffer_bnds = np.array([1, 2])
        R_min = 2
        # pick constant inductance values such that the first LR corner frequency is one decade above f_min
        L0 = R_min / (2 * np.pi * f_bnds[0] * 10 ** decade_buffer_bnds[0])
        L = np.ones(self.N) * L0
        # pick the multiplier for the geometric progression of resistances such that the corner frequencies are equally spaced in log scale between (f_min+1 decade) and (f_max-1 decade)
        R_mult = np.power(
            10,
            (np.log10(f_bnds[1]) - np.log10(f_bnds[0]) - np.sum(decade_buffer_bnds))
            / self.N,
        )
        # geometric progression of resistances
        R = np.array([R_min * R_mult**i for i in range(self.N)])

        param_dict = {"R": R, "L": L}
        return param_dict

    def test_network(self, f_bnds=[1, 1e6]):
        """Test the Cauer RL network synthesis"""
        Nf = 1000
        example_params = self.make_interesting_params(f_bnds=f_bnds)
        self.param_dict["R"] = example_params["R"]
        self.param_dict["L"] = example_params["L"]
        # evaluate and plot the transfer function
        f = np.logspace(np.log10(f_bnds[0]), np.log10(f_bnds[1]), Nf)
        Z = self.eval_Z(f)
        # make state space model
        self.make_state_space()
        # evaluate the state space model in the frequency domain
        sys = ctrl.ss(self.Ay, self.By, self.Cy, self.Dy)
        Y_ss = sys(1j * 2 * np.pi * f)
        Z_ss = 1 / Y_ss

        Z_dict = {"tf": Z, "ss": Z_ss}
        plot_bode_R_L(f, Z_dict)

        self.analyze_numerical_stability(
            test_flux_state=False, test_balanced_state=False
        )

        return f, Z


@register_oneport
class Cauer_RLR(OnePortNetwork):
    def __init__(self, N=3, params=None, T0=0, rho_alpha=0.00393):
        # super().__init__(N)
        self.type = "Cauer_RLR"
        # if params are given in a dictionary make sure keys are 'R' and 'L'
        if params and "R" in params and "L" in params:
            if len(params["R"]) != N + 1 or len(params["L"]) != N:
                raise ValueError("Length of R and L should be equal to order")
            self.param_dict = {"R": params["R"], "L": params["L"]}
            super().__init__(N)
            R_sum = np.sum(self.param_dict["R"])
            self.Tu = 1 / R_sum * np.ones((self.N + 1, 1))
            self.make_state_space()
        else:
            self.param_dict = {"R": np.zeros(N + 1), "L": np.zeros(N)}
            super().__init__(N)
            self.Tu = np.zeros((self.N + 1, 1))

        # define the Tx and Tu matrices for power dissipation
        self.Tx = np.triu(np.ones((self.N + 1, self.N)), 0)
        # the sum of resitances is used for Tu so this need to be updated dynamically

    def __str__(self):
        print_str = ""
        print_str += "Cauer RLR Network\n"
        # add string from parent class
        print_str += super().__str__()
        return print_str

    def pack_params(self, param_dict):
        """make a 1D array of log10 parameters to optimize over"""
        param_arr = np.concatenate(
            (np.log10(param_dict["R"]), np.log10(param_dict["L"]))
        )
        return param_arr

    def unpack_params(self, param_arr):
        """unpack the 1D array of log10 parameters"""
        params = {}
        params["R"] = 10 ** param_arr[: self.N + 1]
        params["L"] = 10 ** param_arr[self.N + 1 :]
        self.param_dict["R"] = params["R"]
        self.param_dict["L"] = params["L"]
        R_sum = np.sum(self.param_dict["R"])
        # update Tu matrix to be the sum of resistances
        self.Tu = 1 / R_sum * np.ones((self.N + 1, 1))

    def estimate_bounds(self, f, Z):
        """estimates conservative bounds for each parameter based on the data"""
        # calculate R and L
        R = np.real(Z)
        L = np.imag(Z) / (2 * np.pi * f)
        L_dc = L[0]
        R_min = np.min(R) / 10
        R_max = 1e3 * np.max(R)
        L_min = 1e-6 * L_dc
        L_max = 1e2 * L_dc

        # assemble the bounds in the for [(min, max)] for each parameter
        bounds = (self.N + 1) * [(R_min, R_max)] + self.N * [(L_min, L_max)]
        # take log10 of every value
        bounds = [(np.log10(b[0]), np.log10(b[1])) for b in bounds]
        return bounds

    def eval_Z(self, f):
        """iteratively add to the one port impedance"""
        s = 1j * 2 * np.pi * f
        R = self.param_dict["R"]
        L = self.param_dict["L"]
        # last RL pair first
        Z = 1 / (1 / R[-1] + 1 / (L[-1] * s))
        # add the effect of the rest of the pairs
        for i in range(self.N - 2, -1, -1):
            Z = 1 / (1 / (L[i] * s) + 1 / (R[i + 1] + Z))
        Z += R[0]
        return Z

    def make_state_space(self):
        """convert RL network to state space"""
        # NOTE: this is still experimental
        N = self.N
        R = self.param_dict["R"]
        L = self.param_dict["L"]
        R_sum = np.sum(R)

        A = np.zeros((N, N))
        B = np.zeros((N, 1))
        for i in range(N):
            B[i, 0] = (1 - np.sum(R[: i + 1]) / R_sum) / L[i]
            for j in range(N):
                A[i, j] = -np.sum(R[: min(i, j) + 1]) / L[i]
        C = np.ones((1, N))
        D = np.ones((1, 1)) / R_sum

        Az, Bz, Cz, Dz = self.invert_state_space(A, B, C, D)

        self.Ay[:] = A
        self.By[:] = B
        self.Cy[:] = C
        self.Dy[:] = D
        self.Az[:] = Az
        self.Bz[:] = Bz
        self.Cz[:] = Cz
        self.Dz[:] = Dz
        # calculate poles and zeros of Z(s)
        self.pz_from_ss()
        print(np.linalg.cond(self.Az))
        print(np.linalg.cond(self.Ay))

    # should be the same as Cauer_RL with one extra resistance
    def make_interesting_params(self, f_bnds=[1, 1e6]):
        """Generate interesting parameters for the RL network.
        Args:
            f_bnds (list): Frequency bounds of intetest
        Returns:
            param_dict: (dict) Dictionary of parameters that yield an interesting response
        """
        decade_buffer_bnds = np.array([1, 2])
        R_min = 2
        # pick constant inductance values such that the first LR corner frequency is one decade above f_min
        L0 = R_min / (2 * np.pi * f_bnds[0] * 10 ** decade_buffer_bnds[0])
        L = np.ones(self.N) * L0
        # pick the multiplier for the geometric progression of resistances such that the corner frequencies are equally spaced in log scale between (f_min+1 decade) and (f_max-1 decade)
        R_mult = np.power(
            10,
            (np.log10(f_bnds[1]) - np.log10(f_bnds[0]) - np.sum(decade_buffer_bnds))
            / self.N,
        )
        # geometric progression of resistances
        R = np.array([R_min * R_mult**i for i in range(self.N + 1)])

        param_dict = {"R": R, "L": L}
        return param_dict

    def test_network(self, f_bnds=[1, 1e6]):
        """Test the Cauer RLR network synthesis"""
        Nf = 1000
        example_params = self.make_interesting_params(f_bnds=f_bnds)
        self.param_dict["R"] = example_params["R"]
        self.param_dict["L"] = example_params["L"]
        # update Tu matrix to be the sum of resistances
        R_sum = np.sum(self.param_dict["R"])
        self.Tu = 1 / R_sum * np.ones((self.N + 1, 1))
        # evaluate and plot the transfer function
        f = np.logspace(np.log10(f_bnds[0]), np.log10(f_bnds[1]), Nf)
        Z = self.eval_Z(f)
        # make state space model
        self.make_state_space()
        # evaluate the state space model in the frequency domain
        sys = ctrl.ss(self.Ay, self.By, self.Cy, self.Dy)
        Y_ss = sys(1j * 2 * np.pi * f)
        Z_ss = 1 / Y_ss
        Z_dict = {"tf": Z, "ss": Z_ss}
        plot_bode_R_L(f, Z_dict)
        self.analyze_numerical_stability(
            test_flux_state=False, test_balanced_state=False
        )
        return f, Z


@register_oneport
class Cauer_LR(OnePortNetwork):
    def __init__(self, N=3, params=None):
        super().__init__(N)
        self.type = "Cauer_LR"
        # if params are given in a dictionary make sure keys are 'R' and 'L' and they are of the correct length
        if params and "R" in params and "L" in params:
            if len(params["R"]) != self.N or len(params["L"]) != self.N:
                raise ValueError("Length of R and L should be equal to order")
            self.param_dict = {"R": params["R"], "L": params["L"]}
            self.make_state_space()
        else:
            self.param_dict = {"R": np.zeros(self.N), "L": np.zeros(self.N)}

        # define the Tx and Tu matrices for power dissipation
        # Tx is lower triangular -1 with remaining values 1 and has size N x N
        self.Tx = np.ones((self.N, self.N))  # Start with all 1s
        self.Tx[np.tril_indices(self.N)] = -1
        self.Tu = np.ones((self.N, 1))

    def __str__(self):
        print_str = ""
        print_str += "Cauer LR Network\n"
        # add string from parent class
        print_str += super().__str__()
        return print_str

    def pack_params(self, param_dict):
        """make a 1D array of log10 parameters to optimize over"""
        param_arr = np.concatenate(
            (np.log10(param_dict["R"]), np.log10(param_dict["L"]))
        )
        return param_arr

    def unpack_params(self, param_arr):
        """unpack the 1D array of log10 parameters"""
        params = {}
        params["R"] = 10 ** param_arr[: self.N]
        params["L"] = 10 ** param_arr[self.N :]
        self.param_dict["R"] = params["R"]
        self.param_dict["L"] = params["L"]

    def estimate_bounds(self, f, Z):
        """estimates conservative bounds for each parameter based on the data"""
        # calculate R and L
        R = np.real(Z)
        L = np.imag(Z) / (2 * np.pi * f)
        L_dc = L[0]
        R_min = np.min(R) * 1e-6  # 10 times smaller than the LF resistance
        R_max = 1e3 * np.max(R)  # 1000 times max real impedance
        L_min = 1e-6 * L_dc
        L_max = 1e2 * L_dc

        # assemble the bounds in the for [(min, max)] for each parameter
        bounds = self.N * [(R_min, R_max)] + self.N * [(L_min, L_max)]

        # take log10 of every value
        bounds = [(np.log10(b[0]), np.log10(b[1])) for b in bounds]

        return bounds

    def eval_Z(self, f):
        """iteratively add to the one port impedance"""
        # TODO: may need to define special case for the case where N=1
        s = 1j * 2 * np.pi * f
        R = self.param_dict["R"]
        L = self.param_dict["L"]
        # last RL pair first
        Z = 1 / (1 / R[-1] + 1 / (L[-1] * s))
        # add the effect of the rest of the pairs
        for i in range(self.N - 2, -1, -1):
            Z = 1 / (1 / (L[i] * s) + 1 / (R[i] + Z))
        return Z

    def make_state_space(self):
        """convert RL network to state space"""
        N = self.N
        R = self.param_dict["R"]
        L = self.param_dict["L"]

        A = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                A[i, j] = -np.sum(R[max(i, j) :]) / L[i]
        B = np.zeros((N, 1))
        for i in range(N):
            B[i, 0] = np.sum(R[i:]) / L[i]
        C = np.ones((1, N))
        for j in range(N):
            C[0, j] = -np.sum(R[j:])
        D = np.sum(R) * np.ones((1, 1))

        Ay, By, Cy, Dy = self.invert_state_space(A, B, C, D)

        self.Az[:] = A
        self.Bz[:] = B
        self.Cz[:] = C
        self.Dz[:] = D
        self.Ay[:] = Ay
        self.By[:] = By
        self.Cy[:] = Cy
        self.Dy[:] = Dy
        # calculate poles and zeros of Z(s)
        self.pz_from_ss()

    def make_interesting_params(self, f_bnds=[1, 1e6]):
        """Generate interesting parameters for the RL network.
        Args:
            f_bnds (list): Frequency bounds of intetest
        Returns:
            param_dict: (dict) Dictionary of parameters that yield an interesting response
        """
        decade_buffer_bnds = np.array([2, 2])
        R_min = 2
        # pick constant inductance values such that the first LR corner frequency is one decade above f_min
        L0 = R_min / (2 * np.pi * f_bnds[0] * 10 ** decade_buffer_bnds[0])
        L = np.ones(self.N) * L0
        # pick the multiplier for the geometric progression of resistances such that the corner frequencies are equally spaced in log scale between (f_min+1 decade) and (f_max-1 decade)
        R_mult = np.power(
            10,
            (np.log10(f_bnds[1]) - np.log10(f_bnds[0]) - np.sum(decade_buffer_bnds))
            / self.N,
        )
        # geometric progression of resistances
        R = np.array([R_min * R_mult**i for i in range(self.N + 1)])

        param_dict = {"R": R, "L": L}
        return param_dict

    def test_network(self, f_bnds=[1, 1e6]):
        """Test the Cauer LR network synthesis"""
        Nf = 1000
        example_params = self.make_interesting_params(f_bnds=f_bnds)
        self.param_dict["R"] = example_params["R"]
        self.param_dict["L"] = example_params["L"]
        # evaluate and plot the transfer function
        f = np.logspace(np.log10(f_bnds[0]), np.log10(f_bnds[1]), Nf)
        Z = self.eval_Z(f)
        # make state space model
        self.make_state_space()
        # evaluate the state space model in the frequency domain
        sys = ctrl.ss(self.Az, self.Bz, self.Cz, self.Dz)
        # Note: unlike RL, the ss model derived was directly for Z not Y
        Z_ss = sys(1j * 2 * np.pi * f)
        Z_dict = {"tf": Z, "ss": Z_ss}
        plot_bode_R_L(f, Z_dict)
        self.analyze_numerical_stability(
            test_flux_state=False, test_balanced_state=False
        )
        return f, Z


# same as cauer RL but with an additional shunt L before the first R, resulting in zero DC impedance and infinite ac impedance
@register_oneport
class Cauer_LRL(OnePortNetwork):
    def __init__(self, N=3, params=None):
        super().__init__(N)
        self.type = "Cauer_LRL"
        # if params are given in a dictionary make sure keys are 'R' and 'L' and they are of the correct length
        if params and "R" in params and "L" in params:
            if len(params["R"]) != self.N or len(params["L"]) != self.N + 1:
                raise ValueError("Length of R or L is incorrect")
            self.param_dict = {"R": params["R"], "L": params["L"]}
            self.make_state_space()
        else:
            self.param_dict = {"R": np.zeros(self.N), "L": np.zeros(self.N + 1)}

        # define the Tx and Tu matrices for power dissipation
        self.Tx = np.triu(np.ones((self.N, self.N)), 0)  # upper triangular ones matrix
        self.Tu = np.zeros((self.N, 1))  # No feedthrough

    def __str__(self):
        print_str = ""
        print_str += "Cauer LRL Network\n"
        # add string from parent class
        print_str += super().__str__()
        return print_str

    def pack_params(self, param_dict):
        """make a 1D array of log10 parameters to optimize over"""
        param_arr = np.concatenate(
            (np.log10(param_dict["R"]), np.log10(param_dict["L"]))
        )
        return param_arr

    def unpack_params(self, param_arr):
        """unpack the 1D array of log10 parameters"""
        params = {}
        params["R"] = 10 ** param_arr[: self.N]
        params["L"] = 10 ** param_arr[self.N :]
        self.param_dict["R"] = params["R"]
        self.param_dict["L"] = params["L"]

    def estimate_bounds(self, f, Z):
        """estimates conservative bounds for each parameter based on the data"""
        # calculate R and L
        R = np.real(Z)
        L = np.imag(Z) / (2 * np.pi * f)
        L_dc = L[0]
        R_min = np.min(R) * 1e-6  # 10 times smaller than the LF resistance
        R_max = 1e3 * np.max(R)  # 1000 times max real impedance
        L_min = 1e-6 * L_dc
        L_max = 1e2 * L_dc

        # assemble the bounds in the for [(min, max)] for each parameter
        bounds = self.N * [(R_min, R_max)] + (self.N + 1) * [(L_min, L_max)]

        # take log10 of every value
        bounds = [(np.log10(b[0]), np.log10(b[1])) for b in bounds]

        return bounds

    def eval_Z(self, f):
        """iteratively add to the one port impedance"""
        s = 1j * 2 * np.pi * f
        R = self.param_dict["R"]
        L = self.param_dict["L"]
        # last RL pair first
        Z = R[-1] + L[-1] * s
        # add the effect of the rest of the pairs
        for i in range(self.N - 2, -1, -1):
            Z = R[i] + 1 / (1 / (L[i + 1] * s) + 1 / Z)
        # add the shunt L
        Z = 1 / (1 / (L[0] * s) + 1 / Z)
        return Z

    def make_state_space(self):
        """convert RL network to state space"""
        N = self.N
        R = self.param_dict["R"]
        L = self.param_dict["L"]
        # The state space matrices will be constructed by modifying a cauer_RL ss matrices omitting L1
        cauer_rl = Cauer_RL(N=self.N, params={"R": R, "L": L[1:]})
        cauer_rl.make_state_space()
        Ay_RL = cauer_rl.Ay
        By_RL = cauer_rl.By
        Cy_RL = cauer_rl.Cy
        L1 = L[0]
        self.Az_s[:] = Ay_RL - L1 / (1 + L1 * Cy_RL @ By_RL) * By_RL @ Cy_RL @ Ay_RL
        self.Bz_s[:] = L1 * By_RL / (1 + L1 * Cy_RL @ By_RL)
        self.Cz_s[:] = -L1 * Cy_RL @ Ay_RL / (1 + L1 * Cy_RL @ By_RL)
        self.Dz_s[:] = L1 / (1 + L1 * Cy_RL @ By_RL)
        # populate inverse system matrices
        Ays, Bys, Cys, Dys = self.invert_state_space(
            self.Az_s, self.Bz_s, self.Cz_s, self.Dz_s
        )
        self.Ays[:] = Ays
        self.Bys[:] = Bys
        self.Cys[:] = Cys
        self.Dys[:] = Dys

        # calculate poles and zeros of Z(s)
        self.pz_from_ss()

    def make_interesting_params(self, f_bnds=[1, 1e6]):
        """Generate interesting parameters for the RL network.
        Args:
            f_bnds (list): Frequency bounds of intetest
        Returns:
            param_dict: (dict) Dictionary of parameters that yield an interesting response
        """
        decade_buffer_bnds = np.array([1, 2])
        R_min = 2
        # pick constant inductance values such that the first LR corner frequency is one decade above f_min
        L0 = R_min / (2 * np.pi * f_bnds[0] * 10 ** decade_buffer_bnds[0])
        L = np.ones(self.N + 1) * L0
        # pick the multiplier for the geometric progression of resistances such that the corner frequencies are equally spaced in log scale between (f_min+1 decade) and (f_max-1 decade)
        R_mult = np.power(
            10,
            (np.log10(f_bnds[1]) - np.log10(f_bnds[0]) - np.sum(decade_buffer_bnds))
            / (self.N - 1),
        )
        # geometric progression of resistances
        R = np.array([R_min * R_mult**i for i in range(self.N)])

        param_dict = {"R": R, "L": L}
        return param_dict

    def test_network(self, f_bnds=[1, 1e6]):
        """Test the Cauer LRL network synthesis"""
        Nf = 1000
        example_params = self.make_interesting_params(f_bnds=f_bnds)
        self.param_dict["R"] = example_params["R"]
        self.param_dict["L"] = example_params["L"]
        # evaluate and plot the transfer function
        f = np.logspace(np.log10(f_bnds[0]), np.log10(f_bnds[1]), Nf)
        Z = self.eval_Z(f)
        # make state space model
        self.make_state_space()
        # # evaluate the state space model in the frequency domain
        sys = ctrl.ss(self.Az_s, self.Bz_s, self.Cz_s, self.Dz_s)
        s = 1j * 2 * np.pi * f
        Z_ss = s * sys(s)
        Z_dict = {"tf": Z, "ss": Z_ss}
        plot_bode_R_L(f, Z_dict)
        self.analyze_numerical_stability(
            test_flux_state=False, test_balanced_state=False
        )
        return f, Z


def OnePort(N, type: str, **kwargs):
    cls = _oneport_registry.get(type)
    if cls is None:
        raise ValueError(
            f"Unknown OnePortNetwork type: {type}. "
            f"Available types: {list(_oneport_registry.keys())}"
        )
    return cls(N=N, **kwargs)


def fit_network_order_freq_domain(
    f, Z, order_arr, network_type="Cauer_RL", alpha=3, opt_kwargs=None
):
    """Sweeps over different network orders and fits the impedance data to the model.
    For each order, the RMSNE (Root Mean Square Normalized Error) of the model is calculated.
    Args:
        f (array): Frequency array
        Z (complex array): Impedance array
        order_arr (array): Array of orders to sweep over
        network_type (str): Type of network to fit. Options are 'Cauer_RL', 'Cauer_RLR', and 'Cauer_LR'
        alpha (float): Weighting factor for low frequency data see: lf_weighting()
        opt_kwargs (dict): Optional arguments for the optimization function
    Returns:
        fit_result (dict): Dictionary of the fit results for each order
            keys: order
            values: dictionary with keys 'Z_fit': [Nf], 'Z_RMSNE': float, 'oneport': OnePortNetwork
    """
    fit_result = {}
    for i, order in enumerate(order_arr):
        if network_type == "Cauer_RL":
            network = Cauer_RL(order)
        elif network_type == "Cauer_LR":
            network = Cauer_LR(order)
        elif network_type == "Cauer_RLR":
            network = Cauer_RLR(order)
        else:
            raise ValueError("Invalid network type")
        network.fit_network(f, Z, alpha=alpha, opt_kwargs=opt_kwargs)
        fit_result[order] = {
            "Z_fit": network.eval_Z(f),
            "Z_RMSNE": np.sqrt(
                network.cost_f(network.pack_params(network.param_dict), f, Z, alpha=0)
            ),
            "oneport": network,
        }

    return fit_result


def plot_bode_R_L(f, Z):
    """Plots Bode magnitude/phase and R/L of a network in a 2x2 grid.
    Args:
        f (array): Frequency array.
        Z (array or dict): Impedance array or dictionary of impedance arrays indexed by label.
    """
    if isinstance(Z, dict):
        fig, ax = plt.subplots(2, 2, figsize=(8, 6))
        fig.suptitle("Network Impedance Analysis")

        for label, Z_val in Z.items():
            R = np.real(Z_val)
            L = np.imag(Z_val) / (2 * np.pi * f)

            # Bode magnitude plot
            ax[0, 0].semilogx(f, np.abs(Z_val), label=label)
            ax[0, 0].set_title("Bode Plot: Magnitude")
            ax[0, 0].set_ylabel("Magnitude (Ohms)")
            ax[0, 0].grid(True, which="both")
            ax[0, 0].set_yscale("log")

            # Bode phase plot
            ax[1, 0].semilogx(f, np.angle(Z_val), label=label)
            ax[1, 0].set_title("Bode Plot: Phase")
            ax[1, 0].set_ylabel("Phase (radians)")
            ax[1, 0].set_xlabel("Frequency (Hz)")
            ax[1, 0].grid(True, which="both")

            # Resistance plot
            ax[0, 1].semilogx(f, np.abs(R), label=label)
            ax[0, 1].set_title("Resistance vs Frequency")
            ax[0, 1].set_ylabel("Resistance (Ohms)")
            ax[0, 1].grid(True, which="both")
            ax[0, 1].set_yscale("log")

            # Inductance plot
            ax[1, 1].semilogx(f, 1e3 * L, label=label)
            ax[1, 1].set_title("Inductance vs Frequency")
            ax[1, 1].set_ylabel("Inductance (mH)")
            ax[1, 1].set_xlabel("Frequency (Hz)")
            ax[1, 1].grid(True, which="both")
            # if np.sign(L).all() == 1:  # if the inductance is all positive
            #     ax[1, 1].set_yscale("log")

        for a in ax.flat:
            a.legend()
    else:
        R = np.real(Z)
        L = np.imag(Z) / (2 * np.pi * f)

        fig, ax = plt.subplots(2, 2, figsize=(8, 6))
        fig.suptitle("Network Impedance Analysis")

        # Bode magnitude plot
        ax[0, 0].semilogx(f, np.abs(Z))
        ax[0, 0].set_title("Bode Plot: Magnitude")
        ax[0, 0].set_ylabel("Magnitude (Ohms)")
        ax[0, 0].grid(True, which="both")
        ax[0, 0].set_yscale("log")

        # Bode phase plot
        ax[1, 0].semilogx(f, np.angle(Z))
        ax[1, 0].set_title("Bode Plot: Phase")
        ax[1, 0].set_ylabel("Phase (radians)")
        ax[1, 0].set_xlabel("Frequency (Hz)")
        ax[1, 0].grid(True, which="both")

        # Resistance plot
        ax[0, 1].semilogx(f, R)
        ax[0, 1].set_title("Resistance vs Frequency")
        ax[0, 1].set_ylabel("Resistance (Ohms)")
        ax[0, 1].grid(True, which="both")
        ax[0, 1].set_yscale("log")

        # Inductance plot
        ax[1, 1].semilogx(f, L)
        ax[1, 1].set_title("Inductance vs Frequency")
        ax[1, 1].set_ylabel("Inductance (H)")
        ax[1, 1].set_xlabel("Frequency (Hz)")
        ax[1, 1].grid(True, which="both")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def compare_meas_fit_RL(f, Z, Z_fit_dict):
    """Plots the measured and fitted impedance data in a variety of formats. The formats include
    - Re(Z) vs frequency with the fit for each order and the fit error in dB
    - L vs frequency with the fit for each order and the fit error in dB
    Args:
        f (np.array) [Nf]: frequency data in Hz
        Z (complex np.array) [Nf]: impedance data
        Z_fit_dict (dict): dictionary of fit data. keys are fit order integers and values are Z_fit arrays
    """
    # make sure there is no megative reactance or resistance
    if np.any(np.real(Z) < 0) or np.any(np.imag(Z) < 0):
        print("Warning: Negative resistance or reactance in measured data")
        return

    order_arr = list(Z_fit_dict.keys())
    order_arr.sort()
    line_colors = plt.cm.viridis(np.linspace(1, 0, len(Z_fit_dict)))
    color_dict = {order: line_colors[i] for i, order in enumerate(order_arr)}

    # first compute the dB error of each fit (use dictionary comprehension)
    R_meas = np.real(Z)
    L_meas = np.imag(Z) / (2 * np.pi * f)
    R_fit_dict = {order: np.real(Z_fit_dict[order]) for order in Z_fit_dict}
    L_fit_dict = {
        order: np.imag(Z_fit_dict[order]) / (2 * np.pi * f) for order in Z_fit_dict
    }
    R_dB_fit_err_dict = {
        order: 20 * np.log10(R_fit_dict[order] / R_meas) for order in Z_fit_dict
    }
    L_dB_fit_err_dict = {
        order: 20 * np.log10(L_fit_dict[order] / L_meas) for order in Z_fit_dict
    }

    # plot Re(Z) above dBerr of fits in two axes figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(1e-3 * f, R_meas, "k--", label="Measured")
    # add each fit
    for order in Z_fit_dict:
        ax[0].plot(
            1e-3 * f,
            R_fit_dict[order],
            label="Order {}".format(order),
            color=color_dict[order],
        )
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlim([1e-3 * f[0], 1e-3 * f[-1]])
    ax[0].set_xlabel("Frequency [kHz]")
    ax[0].set_ylabel("$Re(Z)$ $[\\Omega]$")
    ax[0].grid()
    ax[0].legend()
    ax[0].set_title("Resistance vs Frequency")

    # plot error in dB
    ax[1].plot(1e-3 * f, np.zeros_like(f), "k--", label="Measured")
    for order in Z_fit_dict:
        ax[1].plot(
            1e-3 * f,
            R_dB_fit_err_dict[order],
            label="Order {}".format(order),
            color=color_dict[order],
        )
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Frequency [kHz]")
    ax[1].set_xlim([1e-3 * f[0], 1e-3 * f[-1]])
    ax[1].set_ylabel("Error [dB]")
    ax[1].grid()
    ax[1].legend()
    ax[1].set_title("Resistance Fit Error vs Frequency")
    plt.tight_layout()
    plt.show()

    # plot Inductance and error in dB
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(1e-3 * f, 1e3 * L_meas, "k--", label="Measured")
    for order in Z_fit_dict:
        ax[0].plot(
            1e-3 * f,
            1e3 * L_fit_dict[order],
            label="Order {}".format(order),
            color=color_dict[order],
        )
    ax[0].set_xscale("log")
    ax[0].set_xlim([1e-3 * f[0], 1e-3 * f[-1]])
    ax[0].set_xlabel("Frequency [kHz]")
    ax[0].set_ylabel("Inductance [mH]")
    ax[0].grid()
    ax[0].legend()
    ax[0].set_title("Inductance vs Frequency")

    ax[1].plot(
        1e-3 * f,
        np.zeros_like(f),
        "k--",
        label="Measured",
    )
    for order in Z_fit_dict:
        ax[1].plot(
            1e-3 * f,
            L_dB_fit_err_dict[order],
            label="Order {}".format(order),
            color=color_dict[order],
        )
    ax[1].set_xscale("log")
    ax[1].set_xlim([1e-3 * f[0], 1e-3 * f[-1]])
    ax[1].set_xlabel("Frequency [kHz]")
    ax[1].set_ylabel("Error [dB]")
    ax[1].grid()
    ax[1].legend()
    ax[1].set_title("Inductance Fit Error vs Frequency")
    plt.tight_layout()
    plt.show()


def plot_network_fit_order(fit_result):
    """Plots the RMSNE of the impedance fit for each order.
    Args:
        fit_result (dict): Dictionary of the fit results for each order
            keys: order
            values: dictionary with keys 'Z_fit': [Nf], 'Z_RMSNE': float, 'network_object': OnePortNetwork
    """
    order_arr = list(fit_result.keys())
    order_arr.sort()
    RMSNE_arr = [fit_result[order]["Z_RMSNE"] for order in order_arr]
    # convert to dB
    RMSNE_dB_arr = 20 * np.log10(RMSNE_arr)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(order_arr, RMSNE_dB_arr, "o-")
    ax.set_title("Impedance Fit Error vs Network Order")
    ax.set_xlabel("Network Order")
    ax.set_ylabel("RMSNE")
    ax.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# weighting function for fitting impedance data
def lf_weighting(f, alpha=5):
    """Weighting function for low frequency data points.
    Args:
        f (array): Frequency array.
        alpha (float): Factor to adjust the weighting function relative to the frequency range.
            alpha=1 results in preferential weighting at the lowest frequency or ~3.32
            alpha=10 results in preferential weighting at the lowest frequency or ~24.2
    Returns:
        w (array): Weighting array.
    """
    if alpha == 0:
        w = np.ones_like(f)
    else:
        w = 1 + 1 / np.log10(1 + f / (f[0] * alpha))
        # normalize the weights to sum to avg 1
        w = w / np.mean(w)
    return w


def plot_weighting_function(f_bnds=[1, 1e6], alpha_list=None):
    """Plots a few examples of the weighting function for different alpha values.
    Args:
        f_bnds (list): Frequency bounds of intetest
    """
    if alpha_list is None:
        alpha_list = [0, 1, 2, 5, 10]
    f = np.logspace(np.log10(f_bnds[0]), np.log10(f_bnds[1]), 1000)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_title("Weighting Function for Low Frequency Data")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Weighting")
    ax.grid(True, which="both")
    for alpha in alpha_list:
        w = lf_weighting(f, alpha=alpha)
        ax.semilogx(f, w, label=f"alpha={alpha}")
    ax.legend()
    plt.show()
    plt.tight_layout(rect=[0, 0, 1, 0.96])


def interp_Z(f, Z, N):
    """Interpolate impedance data to a new frequency array.
    Args:
        f (array): Frequency array.
        Z (array): Impedance array.
        N (int): Number of points in the new frequency array.
    Returns:
        f_interp (array): New frequency array.
        Z_interp (array): Interpolated impedance array.
    """
    Nf = len(f)
    R = np.real(Z)
    X = np.imag(Z)
    # make sure X is strictly postive
    if np.min(X) <= 0:
        print("Warning: Imaginary part of impedance is not strictly positive")
        return None
    else:
        R_log = np.log10(R)
        X_log = np.log10(X)
        f_interp = np.logspace(np.log10(f[0]), np.log10(f[-1]), N)
        R_log_interp = np.interp(np.log10(f_interp), np.log10(f), R_log)
        X_log_interp = np.interp(np.log10(f_interp), np.log10(f), X_log)
        Z_interp = 10**R_log_interp + 1j * 10**X_log_interp

    return f_interp, Z_interp
