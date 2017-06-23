import interactpy
import openravepy as orpy
import numpy as np
import pickle
import json
import trajoptpy
from trajoptpy.check_traj import traj_is_safe


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
    robot.SetActiveDOFs(robot.arm.GetArmIndices())
    robot.SetTransform(robot_T)
    robot.SetDOFValues(dofs)
    mug = env.GetKinBody('mug')
    robot.Grab(mug)
    viewer = env.GetViewer()
    viewer.SetCamera(camera_T)
    return env, robot


def get_ik_solns(robot, goal_T):
    with robot.GetEnv():
        ik_param = orpy.IkParameterization(
            goal_T, orpy.IkParameterizationType.Transform6D)
        ik_solutions = robot.arm.FindIKSolutions(
            ik_param, orpy.IkFilterOptions.CheckEnvCollisions,
            ikreturn=False, releasegil=True)
        return ik_solutions


def waypoints_to_traj(env, robot, waypoints):
    traj = orpy.RaveCreateTrajectory(env, '')
    traj.Init(robot.GetActiveConfigurationSpecification('linear'))
    for i in range(waypoints.shape[0]):
        traj.Insert(i, waypoints[i])
    return traj


def trajopt_plan_to_ee(env, robot, goal_T, num_steps=10, num_inits=20):
    local_dir = np.array([0, 0, 1])
    arm_joints = [robot.GetJointFromDOFIndex(idx)
                  for idx in robot.arm.GetArmIndices()]
    def f(x):
        robot.SetActiveDOFValues(x)
        return robot.arm.hand.GetTransform()[:2,:3].dot(local_dir)
    def dfdx(x):
        robot.SetActiveDOFValues(x)
        world_dir = robot.arm.hand.GetTransform()[:3,:3].dot(local_dir)
        return np.array([np.cross(joint.GetAxis(), world_dir)[:2]
                         for joint in arm_joints]).T.copy()

    xyz = goal_T[:3,-1].tolist()
    wxyz = orpy.quatFromRotationMatrix(goal_T[:3,:3]).tolist()
    ik_solns = get_ik_solns(robot, goal_T)
    np.random.shuffle(ik_solns)
    for i in range(min(20, ik_solns.shape[0])):
        init_joint_target = ik_solns[i].tolist()
        request = {
            "basic_info" : {
                "n_steps" : num_steps,
                "manip" : robot.GetName(),
                "start_fixed" : True
            },
            "costs" : [
                {
                    "type" : "joint_vel",
                    "params": {"coeffs" : [1]}
                },
                {
                    "type" : "collision",
                    "params" : {
                        "coeffs" : [20],
                        "dist_pen" : [0.025]
                    }
                }
            ],
            "constraints" : [
                {
                    "type" : "pose",
                    "params" : {
                        "xyz": xyz,
                        "wxyz": wxyz,
                        "link": robot.arm.hand.GetName(),
                        "timestep": num_steps - 1
                    }
                }
            ],
            "init_info" : {
                "type" : "straight_line",
                "endpoint" : init_joint_target
            }
        }
        s = json.dumps(request)
        prob = trajoptpy.ConstructProblem(s, env)
        for t in range(num_steps):
            prob.AddConstraint(f, dfdx, [(t,j) for j in xrange(7)], "EQ", "up%i"%t)
        result = trajoptpy.OptimizeProblem(prob)
        is_safe = traj_is_safe(
            result.GetTraj(), robot)
        if is_safe:
            return waypoints_to_traj(env, robot, result.GetTraj())
    print('Failed to solve.')


