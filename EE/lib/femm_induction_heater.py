import femm
import numpy as np
import os
import yaml
import dill as pickle
import time


"""
Functions for interfacing with the FEMM software for simulating magnetic components (both electrostatic and magnetic)
"""


def init_coil_magnetic_problem(f=0.0, hide_window=True):
    """initializes an ac magnetics problem in femm for testing a two winding system with the
    a primary coil with a single secondary coil loaded with a defined load impedance
    Args
        f: frequency of the problem [Hz]
        hide_window: if True, femm will be run in the background
    """
    femm.openfemm(hide_window)
    femm.newdocument(0)  # 0 for magentics
    femm.mi_probdef(f, "meters", "axi", 1e-8, 0, 30)

    # add materials
    femm.mi_getmaterial("Air")
    femm.mi_getmaterial("Copper")
    # add graphite and other materials

    I_port = 1.0
    femm.mi_addcircprop("port1", I_port, 1)


def init_injection_xfmr_magnetic_problem(xfmr_config, f=0.0, hide_window=True):
    """initializes an ac magnetics problem in femm for testing a toroidal injection tranformer with a centeral axial single turn secondary
    Args
        xfmr_config: dictionary of transformer dimensions, materials, and other parameters
        f: frequency of the problem [Hz]
        hide_window: if True, femm will be run in the background
    """
    femm.openfemm(hide_window)
    femm.newdocument(0)  # 0 for magentics
    depth = xfmr_config["core"]["h"] * xfmr_config["core"]["N_cores"]
    femm.mi_probdef(f, "meters", "planar", 1e-8, depth, 30)

    # add materials
    femm.mi_getmaterial("Air")
    femm.mi_getmaterial("Copper")
    # get soft ferrite material
    femm.mi_getmaterial("Soft magnetic ferrite")  # Soft magnetic ferrite (Fe-Ni-Zn-V)

    # make a new material with specific material properties (rho_e, mu_r, etc)

    I_port = 1.0
    femm.mi_addcircprop("Pri", I_port, 1)  # series
    femm.mi_addcircprop("Sec", 0.0, 1)  # open circuit


def arrange_windings_coil(coil_config):
    """does all the math for determining the location of each turn based on the
    winding configs
    Args:
        coil_config: dictionary of coil dimensions, materials, and other parameters
    Returns
        RZ_center: [N x 2] array of the (r,z) coordinates of the center of each
            turn in winding order
    """
    winding_r = coil_config["coil"]["winding_r"]
    winding_pitch = coil_config["coil"]["winding_pitch"]
    N_turns = coil_config["coil"]["N_turns"]

    # single layer coil centered at z=0
    z_bnds = np.array(
        [-(N_turns - 1) / 2 * winding_pitch, (N_turns - 1) / 2 * winding_pitch]
    )
    z_arr = np.linspace(z_bnds[0], z_bnds[1], N_turns)
    RZ_center = np.stack((np.full(N_turns, winding_r), z_arr), axis=-1)  # [N x 2]
    return RZ_center


def arrange_windings_toroid(xfmr_config):
    """does all the math for determining the location of each turn based on the
    winding configs for a toroidal transformer
    Args:
        xfmr_config: dictionary of transformer dimensions, materials, and other parameters
    Returns
        pri_rz: (r,z) coordinates of the center of the primary turns
        sec_rz: (r,z) coordinates of the center of the secondary turn
    """
    sec_rz = (0, 0)
    core_ri = xfmr_config["core"]["ri"]
    core_ro = xfmr_config["core"]["ro"]
    bobbin_dr = xfmr_config["primary"]["bobbin_dr"]
    wire_d = xfmr_config["primary"]["wire_d"]
    Np = xfmr_config["primary"]["N"]
    winding_r_inner = core_ri - bobbin_dr - wire_d / 2
    winding_r_outer = core_ro + bobbin_dr + wire_d / 2
    # windings are evenly spaced in angle
    theta_arr = np.linspace(0, 2 * np.pi, Np, endpoint=False)
    # make arrays of inner and outer primary rz coords
    # TODO
    return None


# core:
#   ri: 27.0e-3  # inner radius
#   ro: 42.0e-3  # outer radius
#   h: 12.7e-3  # core height
#   Al: 2726e-9  # [H/T^2]  # inductance factor
#   N_cores: 2  # number of cores stacked in axial direction
# secondary:
#   tube_d: 12.7e-3  # copper tube OD
#   tube_t: 0.7e-3  # copper tube thickness
# primary:
#   bobbin_dr: 1.0e-3  # radial thickness of the bobbin
#   wire_d: 3.0e-3  # diameter of the wire
#   litz_strands: 100  # number of strands in litz wire per conductor
#   N: 10  # number of turns


def make_hollow_rectangular_turn(
    turn_center,
    coil_config,
    circuit_name="<None>",
    skin_depth=None,
):
    """makes a single turn of a hollow rectangular conductor for magnetic simulation
    Args:
        turn_center: (r,z) coordinates of the center of the turn
        coil_config: dictionary of coil parameters
        circuit_name: name of the circuit for the conductor
        skin_depth: skin depth of the conductor material at the simulation frequency [m]
    """
    dr = coil_config["coil"]["tube_dr"]
    dz = coil_config["coil"]["tube_dz"]
    tube_t = coil_config["coil"]["tube_t"]  # shell thickness
    corner_radius = coil_config["coil"]["tube_radius"]

    # crossection mesh size
    min_dim = min(dr, dz)
    mesh_size_factor = coil_config["coil"].get(
        "mesh_size_factor", 10
    )  # default factor of 10
    mesh_size = min_dim / mesh_size_factor
    # skin mesh size
    skin_mesh_size = np.min([skin_depth / 2, mesh_size])
    arc_geometry_max_seg_deg = 10  # degrees

    # Calculate safe corner radius first
    max_outer_radius = min(dr, dz) / 2
    safe_corner_radius = min(corner_radius, max_outer_radius)
    mesh_max_seg_deg = (
        360 * skin_mesh_size / (2 * safe_corner_radius * np.pi)
        if safe_corner_radius > 0
        else 10
    )

    # Outer rectangle coordinates (ul, ur, lr, ll)
    outer_corner_coords = [
        (turn_center[0] - dr / 2, turn_center[1] + dz / 2),
        (turn_center[0] + dr / 2, turn_center[1] + dz / 2),
        (turn_center[0] + dr / 2, turn_center[1] - dz / 2),
        (turn_center[0] - dr / 2, turn_center[1] - dz / 2),
    ]

    # Inner rectangle coordinates (smaller by tube_t on all sides)
    inner_corner_coords = [
        (turn_center[0] - dr / 2 + tube_t, turn_center[1] + dz / 2 - tube_t),
        (turn_center[0] + dr / 2 - tube_t, turn_center[1] + dz / 2 - tube_t),
        (turn_center[0] + dr / 2 - tube_t, turn_center[1] - dz / 2 + tube_t),
        (turn_center[0] - dr / 2 + tube_t, turn_center[1] - dz / 2 + tube_t),
    ]

    # Draw outer rectangle
    femm.mi_drawrectangle(*outer_corner_coords[3], *outer_corner_coords[1])

    # Draw inner rectangle (hole)
    femm.mi_drawrectangle(*inner_corner_coords[3], *inner_corner_coords[1])

    # Add radius to the corners to improve meshing
    # Use the safe corner radius calculated above

    if safe_corner_radius > 0.1e-3:  # Only create if radius is meaningful (> 0.1mm)
        # Inner corners - calculate safe radius FIRST before modifying outer corners
        # Maximum possible inner radius is limited by the inner rectangle dimensions
        inner_dr = dr - 2 * tube_t  # inner rectangle width
        inner_dz = dz - 2 * tube_t  # inner rectangle height

        # Ensure inner corner radius doesn't exceed half the smallest inner dimension
        max_inner_radius = min(inner_dr, inner_dz) / 2
        inner_corner_radius = max(0, min(safe_corner_radius - tube_t, max_inner_radius))

        # Create inner corner radii FIRST (before outer corners modify the geometry)
        if (
            inner_corner_radius > 0.1e-3
            and inner_dr > 2 * inner_corner_radius
            and inner_dz > 2 * inner_corner_radius
        ):
            for corner_coord in inner_corner_coords:
                femm.mi_createradius(*corner_coord, inner_corner_radius)

        # Now create outer corners
        for corner_coord in outer_corner_coords:
            femm.mi_createradius(*corner_coord, safe_corner_radius)

        # Set arc properties for outer corners
        outer_ul_coord = (
            outer_corner_coords[0][0] + safe_corner_radius,
            outer_corner_coords[0][1] - safe_corner_radius,
        )
        outer_ur_coord = (
            outer_corner_coords[1][0] - safe_corner_radius,
            outer_corner_coords[1][1] - safe_corner_radius,
        )
        outer_lr_coord = (
            outer_corner_coords[2][0] - safe_corner_radius,
            outer_corner_coords[2][1] + safe_corner_radius,
        )
        outer_ll_coord = (
            outer_corner_coords[3][0] + safe_corner_radius,
            outer_corner_coords[3][1] + safe_corner_radius,
        )

        femm.mi_selectarcsegment(*outer_ul_coord)
        femm.mi_selectarcsegment(*outer_ur_coord)
        femm.mi_selectarcsegment(*outer_lr_coord)
        femm.mi_selectarcsegment(*outer_ll_coord)

        # Set arc properties for inner corners if they exist
        if (
            inner_corner_radius > 0.1e-3
            and inner_dr > 2 * inner_corner_radius
            and inner_dz > 2 * inner_corner_radius
        ):
            inner_ul_coord = (
                inner_corner_coords[0][0] - inner_corner_radius,
                inner_corner_coords[0][1] + inner_corner_radius,
            )
            inner_ur_coord = (
                inner_corner_coords[1][0] + inner_corner_radius,
                inner_corner_coords[1][1] + inner_corner_radius,
            )
            inner_lr_coord = (
                inner_corner_coords[2][0] + inner_corner_radius,
                inner_corner_coords[2][1] - inner_corner_radius,
            )
            inner_ll_coord = (
                inner_corner_coords[3][0] - inner_corner_radius,
                inner_corner_coords[3][1] - inner_corner_radius,
            )

            femm.mi_selectarcsegment(*inner_ul_coord)
            femm.mi_selectarcsegment(*inner_ur_coord)
            femm.mi_selectarcsegment(*inner_lr_coord)
            femm.mi_selectarcsegment(*inner_ll_coord)

        femm.mi_setarcsegmentprop(mesh_max_seg_deg, "<None>", 0, 0)
        femm.mi_clearselected()

        # Assign conductor faces to boundary condition (outer edges only)
        femm.mi_selectsegment(turn_center[0], turn_center[1] + dz / 2)
        femm.mi_selectsegment(turn_center[0], turn_center[1] - dz / 2)
        femm.mi_selectsegment(turn_center[0] - dr / 2, turn_center[1])
        femm.mi_selectsegment(turn_center[0] + dr / 2, turn_center[1])
        femm.mi_setsegmentprop("<None>", skin_mesh_size, 0, 0, 0)
        femm.mi_clearselected()

    # Add copper label in the conductor region (between outer and inner rectangles)
    # Place label at a point that's definitely in the copper region
    copper_label_pos = (turn_center[0] - dr / 2 + tube_t / 2, turn_center[1])
    femm.mi_addblocklabel(*copper_label_pos)
    femm.mi_selectlabel(*copper_label_pos)
    femm.mi_setblockprop("Copper", 0, mesh_size, circuit_name, 0, 0, 1)
    femm.mi_clearselected()

    # Add air label in the center (hollow part)
    air_label_pos = turn_center
    femm.mi_addblocklabel(*air_label_pos)
    femm.mi_selectlabel(*air_label_pos)
    femm.mi_setblockprop("Air", 0, mesh_size, "<None>", 0, 0, 1)
    femm.mi_clearselected()


def make_coil_windings_magnetic(coil_config, turn_rz, skin_depth=1.0):
    """makes the geometry for the inductor winding within the core winding area for a
    magnetic simulation
    Args
        coil_config: dictionary of coil parameters
        turn_rz: [N x 2] array of the (r,z) coordinates of the center of each turn
    """
    # make each conductor crossection
    for turn_idx, rz in enumerate(turn_rz):
        # circular crossection
        make_hollow_rectangular_turn(
            rz,
            coil_config,
            circuit_name="port1",
            skin_depth=skin_depth,
        )
    # femm.mi_zoomnatural()


def make_crucible_magnetic(coil_config, include_crucible=False):
    """makes the crucible geometry for a magnetic simulation
    Args
        coil_config: dictionary of coil parameters
    """
    sim_r_max = coil_config["sim_r_max"]
    crucible_r_inner = coil_config["crucible"]["r_i"]
    crucible_r_outer = coil_config["crucible"]["r_o"]
    crucible_height = coil_config["crucible"]["h"]

    if include_crucible:
        # draw crucible as a rectangle
        femm.mi_drawrectangle(
            crucible_r_inner,
            -crucible_height / 2,
            crucible_r_outer,
            crucible_height / 2,
        )
        femm.mi_addblocklabel(
            (crucible_r_inner + crucible_r_outer) / 2, 0
        )  # center of crucible
        femm.mi_selectlabel((crucible_r_inner + crucible_r_outer) / 2, 0)
        femm.mi_setblockprop("Graphite", 0, 0.01, "<None>", 0, 0, 1)
        femm.mi_clearselected()

    # add air label 5mm above the crucible at r = crucible_r_inner/2
    air_label_rz = (sim_r_max, 0)
    femm.mi_addblocklabel(*air_label_rz)
    femm.mi_selectlabel(*air_label_rz)
    femm.mi_setblockprop("Air", 0, 0.01, "<None>", 0, 0, 1)
    femm.mi_clearselected()

    femm.mi_zoomnatural()
    femm.mi_makeABC()


def coil_impedance_sim(coil_config, f_arr, hide_window=False):
    """calculates the inductive impedance of a coil
    Args
        coil_config: dictionary of coil dimensions, materials, and other parameters
        f_arr: frequency array of the problem [Hz]
        hide_window: if True, femm will be run in the background
    Returns
        Z_arr: complex float array impedance of the coil [Ohm]
        sim_info_list: list of simulation information dictionaries
    """
    Z_arr = np.zeros(len(f_arr), dtype=complex)
    sim_info_list = []  # list of simulation information dictionaries

    # simulate the impedance at each frequency
    for f_idx, f in enumerate(f_arr):
        init_coil_magnetic_problem(f=f, hide_window=hide_window)

        # get turn positions using available function
        turn_rz = arrange_windings_coil(coil_config)

        # calculate skin depth (simple approximation)
        # skin depth = sqrt(2*rho/(omega*mu)) where rho=1.7e-8 for copper, mu=4*pi*1e-7
        rho_copper = 1.7e-8  # ohm*m
        mu_0 = 4 * np.pi * 1e-7  # H/m
        omega = 2 * np.pi * f
        if f > 0:
            skin_depth = np.sqrt(2 * rho_copper / (omega * mu_0))
        else:
            skin_depth = 1e-3  # 1mm default for DC

        # make coil windings using available function
        make_coil_windings_magnetic(coil_config, turn_rz, skin_depth=skin_depth)

        make_crucible_magnetic(coil_config, include_crucible=False)

        # save temporary file
        temp_filename = f"coil_sim.fem"
        femm.mi_saveas(temp_filename)

        start_time = time.time()
        femm.mi_analyze()
        end_time = time.time()
        sim_dur = np.around(end_time - start_time, 1)
        femm.mi_loadsolution()
        i, v, flux = femm.mo_getcircuitproperties("port1")

        # avoid division by zero
        if abs(i) > 1e-12:
            Z_arr[f_idx] = v / i
        else:
            Z_arr[f_idx] = complex(np.inf, 0)

        # get the number of elements and nodes
        num_nodes = femm.mo_numnodes()
        num_elements = femm.mo_numelements()

        sim_info_list.append(
            {
                "frequency": f,
                "sim_dur": sim_dur,
                "num_nodes": num_nodes,
                "num_elements": num_elements,
            }
        )

        # close solution and clean up
        femm.mo_close()
        femm.mi_close()

    return Z_arr, sim_info_list
