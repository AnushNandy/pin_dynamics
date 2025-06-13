# src/kinematics/urdf_parser.py
import xml.etree.ElementTree as ET
import numpy as np
import os
import MathUtils
import sys

def parse_vector(xyz_str):
    """ Parses a string 'x y z' into a numpy array [x, y, z]. """
    return np.array([float(s) for s in xyz_str.split()])


def parse_urdf_dynamics(urdf_file_path, target_link_names):
    """
    Parses a URDF file to extract dynamic parameters (mass, CoM, inertia tensor)
    for specified links.

    Args:
        urdf_file_path (str): The full path to the URDF file.
        target_link_names (list[str]): A list of link names for which to extract
                                       dynamic parameters.

    Returns:
        dict: A dictionary where keys are link names and values are dicts
              containing 'mass', 'com' (center of mass relative to link frame),
              and 'inertia_tensor' (3x3 matrix about the CoM, in link frame).
              Returns None if parsing fails.
    """
    try:
        tree = ET.parse(urdf_file_path)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Error: URDF file not found at {urdf_file_path}")
        return None
    except ET.ParseError:
        print(f"Error: Could not parse URDF file {urdf_file_path}. Check XML validity.")
        return None

    parsed_dynamics = {}

    for link_element in root.findall('link'):
        link_name = link_element.get('name')

        if link_name not in target_link_names:
            continue

        inertial_element = link_element.find('inertial')
        if inertial_element is None:
            print(f"Warning: Link '{link_name}' has no <inertial> tag. Assuming massless or placeholder.")
            parsed_dynamics[link_name] = {
                'mass': 0.0,
                'com': np.array([0.0, 0.0, 0.0]),  # Relative to link frame
                'inertia_tensor': np.zeros((3, 3))  # About CoM, in link frame
            }
            continue

        mass_el = inertial_element.find('mass')
        origin_el = inertial_element.find('origin')
        inertia_el = inertial_element.find('inertia')

        if mass_el is None or origin_el is None or inertia_el is None:
            print(f"Warning: Link '{link_name}' has an incomplete <inertial> tag. Skipping dynamics for this link.")
            continue

        mass = float(mass_el.get('value'))
        com_origin = parse_vector(origin_el.get('xyz'))

        ixx = float(inertia_el.get('ixx'))
        ixy = float(inertia_el.get('ixy'))
        ixz = float(inertia_el.get('ixz'))
        iyy = float(inertia_el.get('iyy'))
        iyz = float(inertia_el.get('iyz'))
        izz = float(inertia_el.get('izz'))

        inertia_tensor = np.array([
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],  # Iyx = Ixy
            [ixz, iyz, izz]  # Izx = Ixz, Izy = Iyz
        ])

        parsed_dynamics[link_name] = {
            'mass': mass,
            'com': com_origin,
            'inertia_tensor': inertia_tensor
        }
    # Ensure all target links were found
    for target_name in target_link_names:
        if target_name not in parsed_dynamics:
            print(f"Warning: Target link '{target_name}' not found in URDF or has no inertial properties.")
            # Assign default zero dynamics if critical for downstream consistency
            if target_name.lower() == "base" or "baseframe" in target_name.lower() and target_name not in parsed_dynamics:
                parsed_dynamics[target_name] = {
                    'mass': 0.0,  # Base is often fixed and its mass isn't part of moving chain dynamics for RNEA
                    'com': np.array([0.0, 0.0, 0.0]),
                    'inertia_tensor': np.zeros((3, 3))
                }

    return parsed_dynamics


def generate_config_string(dynamic_params_dict):
    """
    Generates a Python dictionary string for robot_config.py.
    """
    output_str = "LINK_DYNAMIC_PARAMETERS = {\n"
    for link_name, params in dynamic_params_dict.items():
        output_str += f"    \"{link_name}\": {{\n"
        output_str += f"        'mass': {params['mass']},\n"
        output_str += f"        'com': np.array([{params['com'][0]}, {params['com'][1]}, {params['com'][2]}]),\n"
        output_str += f"        'inertia_tensor': np.array([\n"
        output_str += f"            [{params['inertia_tensor'][0, 0]}, {params['inertia_tensor'][0, 1]}, {params['inertia_tensor'][0, 2]}],\n"
        output_str += f"            [{params['inertia_tensor'][1, 0]}, {params['inertia_tensor'][1, 1]}, {params['inertia_tensor'][1, 2]}],\n"
        output_str += f"            [{params['inertia_tensor'][2, 0]}, {params['inertia_tensor'][2, 1]}, {params['inertia_tensor'][2, 2]}]\n"
        output_str += f"        ])\n"
        output_str += f"    }},\n"
    output_str += "}\n"
    return output_str


if __name__ == "__main__":

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))

    URDF_RELATIVE_PATH = r"ArmModels\urdfs\P4\P4_Contra-Angle_right.urdf"
    full_urdf_path = os.path.join(project_root_dir, URDF_RELATIVE_PATH)
    config_dir = os.path.join(project_root_dir, 'config')
    if config_dir not in sys.path:
        sys.path.append(config_dir)

    try:
        import robot_config

        LINK_NAMES_TO_PARSE = robot_config.LINK_NAMES_IN_KDL_ORDER
        print(f"Successfully imported LINK_NAMES_IN_KDL_ORDER from robot_config.py: {LINK_NAMES_TO_PARSE}")
    except ImportError:
        print("Could not import robot_config.py. Using default link names.")
        # These should match your robot_config.LINK_NAMES_IN_KDL_ORDER
        LINK_NAMES_TO_PARSE = [
            "Base", "Link_0", "Link_1", "Link_2",
            "Link_3", "Link_4", "Link_5", "End_Effector"
        ]

    print(f"Looking for URDF at: {full_urdf_path}")
    if not os.path.exists(full_urdf_path):
        print("URDF file not found. Please check the path.")
    else:
        print("URDF file found. Parsing...")
        dynamic_parameters = parse_urdf_dynamics(full_urdf_path, LINK_NAMES_TO_PARSE)

        if dynamic_parameters:
            print("\nSuccessfully parsed dynamic parameters:")
            for link, params in dynamic_parameters.items():
                print(f"  Link: {link}")
                print(f"    Mass: {params['mass']}")
                print(f"    CoM: {params['com']}")
                print(f"    Inertia Tensor:\n{params['inertia_tensor']}")

            print("\n--- Python dictionary string for robot_config.py ---")
            config_output = generate_config_string(dynamic_parameters)
            print(config_output)

            # You can save this to a file or copy-paste it
            output_file_path = os.path.join(current_script_dir, "parsed_dynamic_params.txt")
            with open(output_file_path, "w") as f:
                f.write(config_output)
            print(f"Configuration string saved to: {output_file_path}")
        else:
            print("Failed to parse dynamic parameters.")