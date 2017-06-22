import interactpy
import openravepy as orpy
import numpy as np
import pickle


dofs = np.array([-1.00000003e+00,
                 -3.12429069e+00,
                 -2.98959332e-07,
                 1.64363035e+00,
                 3.25407658e-07,
                 -3.09532854e+00,
                 -1.63913051e-07,
                 7.00000031e-01,
                 5.20000130e-01,
                 6.50000001e-01])
robot_T = np.array([[4.99413199e-01, 8.66363929e-01, 1.03278612e-07, 1.21931720e+00],
                    [-8.66363929e-01, 4.99413199e-01, -5.96745968e-08, -4.44620043e-01],
                    [-1.03278620e-07, -5.96745829e-08, 1.00000000e+00, 5.90718985e-01],
                    [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
camera_T = np.array([[-0.99957573,  0.0139481 ,  0.02556982,  1.23822057],
                     [-0.01719667,  0.42592924, -0.90459304,  1.93561268],
                     [-0.02350829, -0.90464896, -0.42550867,  1.7823844 ],
                     [ 0.        ,  0.        ,  0.        ,  1.        ]])
goal_T = np.array([[ 0.01297332,  0.99651909,  0.08234919,  1.21825334],
                   [-0.00482776,  0.08241758, -0.99658619,  0.21255009],
                   [-0.99990419,  0.01253147,  0.00588019,  0.61503719],
                   [ 0.        ,  0.        ,  0.        ,  1.        ]])


def setup():
    env, robot = interactpy.initialize(saved_env='./default.env.xml')
    robot.SetTransform(robot_T)
    robot.SetDOFValues(dofs)
    mug = env.GetKinBody('mug')
    robot.Grab(mug)
    viewer = env.GetViewer()
    viewer.SetCamera(camera_T)
    return env, robot
