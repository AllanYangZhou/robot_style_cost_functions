import numpy as np
from catkin.find_in_workspaces import find_in_workspaces


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
starting_angles = np.array([
    335.183105469,
    167.792098999,
    598.978515625,
    114.580368042,
    263.228393555,
    264.061767578,
    241.058807373
])
starting_angles = (np.pi / 180.) * starting_angles
starting_angles[2] += np.pi # bug in recording process

iact_ctrl_path = find_in_workspaces(
    project='iact_control',
    path='src/data',
    first_match_only=True)[0]

starting_finger_angles = [.4,.5,.5]

goal_angles = np.array([
    353.210235596,
    241.983413696,
    372.164093018,
    123.811828613,
    357.25100708,
    235.245880127,
    221.578979492
])
goal_angles = (np.pi / 180.) * goal_angles
goal_angles[2] += np.pi
