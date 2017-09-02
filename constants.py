import numpy as np
from catkin.find_in_workspaces import find_in_workspaces


def convert_angles(angles):
    '''Convert angles recorded from physical robot
    to openrave joint angles.
    '''
    converted = (np.pi / 180.) * angles
    converted[2] += np.pi #bug in the recording process
    return converted


camera_T = np.array([[-0.94278671, -0.0492597 ,  0.32973732, -0.98667234],
                     [-0.32941772,  0.28997609, -0.89855319,  2.01822257],
                     [-0.05135347, -0.95576532, -0.28961262,  1.00309241],
                     [ 0.        ,  0.        ,  0.        ,  1.        ]])

table_T = np.array([[0.0, 1.0,  0.0, -0.8128/2],
                    [1.0, 0.0,  0.0, 0],
                    [0.0, 0.0,  1.0, -0.1143],
                    [0.0, 0.0, 0.0, 1.0]])

table_color = np.array([0.9, 0.75, 0.75])

box_T = np.array([[ -3.09204759e-01,  -9.50995487e-01,  -1.13367494e-07,
                    5.28360605e-01],
                  [  9.50995487e-01,  -3.09204759e-01,  -1.56069336e-07,
                     1.39327019e-01],
                  [  1.13367465e-07,  -1.56069357e-07,   1.00000000e+00,
                     7.23589838e-01],
                  [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                     1.00000000e+00]])

cabinet_T = np.array([[  0.00000000e+00,  -1.00000000e+00,   0.00000000e+00,
                         6.02496743e-01],
                      [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                         4.34175527e-05],
                      [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
                         -1.72486659e-02],
                      [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                         1.00000000e+00]])

cabinet_color = np.array([0.05,0.6,0.3])

# Angles recorded from physical robot (degrees)
configs = []
configs.append(convert_angles(np.array([
    335.183105469,
    167.792098999,
    598.978515625,
    114.580368042,
    263.228393555,
    264.061767578,
    241.058807373
])))
configs.append(convert_angles(np.array([
    253.225967407,
    153.073532104,
    202.243759155,
    65.9272537231,
    168.288208008,
    229.51335144,
    167.845901489
])))
configs.append(convert_angles(np.array([
    181.138198853,
    180.720993042,
    95.5344772339,
    71.0014724731,
    171.063491821,
    191.364212036,
    222.647033691
])))
configs.append(convert_angles(np.array([
    353.210235596,
    241.983413696,
    372.164093018,
    123.811828613,
    357.25100708,
    235.245880127,
    221.578979492
])))

iact_ctrl_path = find_in_workspaces(
    project='iact_control',
    path='src/data',
    first_match_only=True)[0]

starting_finger_angles = [.4,.5,.5]

wp_high = np.array([ 6.16468156,  2.9901967 ,  9.63708143,  2.96758199,  6.23520633,
        3.06439853,  3.86728277])
