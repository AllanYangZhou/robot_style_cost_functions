import numpy as np
from catkin.find_in_workspaces import find_in_workspaces
import itertools


def convert_angles(angles):
    '''Convert angles recorded from physical robot
    to openrave joint angles.
    '''
    converted = (np.pi / 180.) * angles
    converted[2] += np.pi #bug in the recording process
    converted = np.mod(converted, 2 * np.pi)
    return converted


camera_T = np.array([[-0.94278671, -0.0492597 ,  0.32973732, -0.98667234],
                     [-0.32941772,  0.28997609, -0.89855319,  2.01822257],
                     [-0.05135347, -0.95576532, -0.28961262,  1.00309241],
                     [ 0.        ,  0.        ,  0.        ,  1.        ]])

two_robot_camera_T = np.array([[-0.94278671, -0.04925967,  0.32973734, -0.96558571],
                               [-0.32941775,  0.2899759 , -0.89855324,  2.99906778],
                               [-0.05135344, -0.95576538, -0.28961244,  0.78374684],
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
    353.210235596,
    241.983413696,
    372.164093018,
    123.811828613,
    357.25100708,
    235.245880127,
    221.578979492
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
    157.691741943,
    105.630950928,
    173.614852905,
    79.6633911133,
    163.452987671,
    252.40512085,
    167.846038818
])))
configs.append(convert_angles(np.array([
    202.639007568,
    111.832611084,
    170.277252197,
    88.8450317383,
    163.454544067,
    239.178985596,
    167.846176147
])))
configs.append(convert_angles(np.array([
    157.010559082,
    173.968551636,
    197.956741333,
    109.312904358,
    153.473571777,
    165.363220215,
    167.844192505
])))
configs.append(convert_angles(np.array([
    183.63142395,
    178.446975708,
    211.959579468,
    87.1731567383,
    138.10369873,
    191.371032715,
    222.647033691
])))
configs.append(np.array([
    -0.61847021,
    3.036324,
    0.31340425,
    1.66226363,
    2.67861919,
    2.88613266,
    2.92943373
]))

sg_pair_idcs = [
    (0,1),
    (1,0),
    (2,3),
    (3,2)
]

sg_test_idcs = [
    (6,8),
    (8,6)
]

start_goal_pairs = []
for i, j in sg_pair_idcs:
    start_goal_pairs.append((configs[i], configs[j]))
# for s, g in itertools.permutations(configs, 2):
#     start_goal_pairs.append((s, g))


iact_ctrl_path = find_in_workspaces(
    project='iact_control',
    path='src/data',
    first_match_only=True)[0]

starting_finger_angles = [.4,.5,.5]

'''Min/max world features from 100k samples.'''
world_feature_min = np.array([-0.0016 , -0.0016 , 0.27549997,
                              -0.20500198, -0.20499972, 0.0705 , -0.4099935 , -0.40999556, -0.1345,
                              -0.61737883, -0.61673892, -0.34186703, -0.72111118, -0.72012073,
                              -0.44557309, -0.81402427, -0.81276798, -0.54588711, -0.10374682,
                              -0.10374886, -0.10374977], dtype=np.float32)
world_feature_max = np.array([ 0.0016 , 0.0016 , 0.27550003,
                               0.20498267, 0.20500229, 0.48050001,
                               0.40997422, 0.40998793, 0.68550003,
                               0.61670083, 0.61614335, 0.8928228 ,
                               0.72012377, 0.71954221, 0.99655741,
                               0.81451184, 0.81669331, 1.09977067,
                               0.10374734, 0.10374925, 0.1037479 ],
                             dtype=np.float32)
world_frange = world_feature_max - world_feature_min
world_frange[2] = 1.0 # prevents divide by 0
