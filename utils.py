import trajoptpy
from trajoptpy.check_traj import traj_is_safe
import interactpy
import openravepy as orpy
import prpy
import numpy as np
import pickle
import json


dofs = np.array([ -1.00000003e+00,  -3.12429069e+00,   1.56311539e+00,
                  1.64363038e+00,   3.77923758e-07,  -3.09532846e+00,
                  -1.64521968e-07,   .6,   .4,   .3])
robot_T = np.array([[  8.41128138e-01,  -5.40835886e-01,  -6.44726267e-08,
          1.22802639e+00],
       [  5.40835886e-01,   8.41128138e-01,  -1.89389699e-08,
         -4.89164531e-01],
       [  6.44726150e-08,  -1.89390097e-08,   1.00000000e+00,
          5.47507167e-01],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          1.00000000e+00]])
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
    box = env.GetKinBody('box1')
    robot.Grab(box)
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
    orpy.planningutils.RetimeActiveDOFTrajectory(
        traj, robot)
    return traj


def make_orientation_cost(robot):
    local_dir = np.array([1, 0, 0])
    arm_joints = [robot.GetJointFromDOFIndex(idx)
                  for idx in robot.arm.GetArmIndices()]
    def f(x):
        robot.SetActiveDOFValues(x)
        return robot.arm.hand.GetTransform()[:3,:3].dot(local_dir) - np.array([0, 0, -1])
    def dfdx(x):
        robot.SetActiveDOFValues(x)
        world_dir = robot.arm.hand.GetTransform()[:3,:3].dot(local_dir)
        return np.array([np.cross(joint.GetAxis(), world_dir)[:3]
                         for joint in arm_joints]).T.copy()
    return f, dfdx


def trajopt_plan_to_ee(env, robot, goal_T,
                       num_steps=10, num_inits=20,
                       w=[1, 0, 0]):
    def f_ee_height(x):
        robot.SetActiveDOFValues(x)
        return w[1] * (robot.arm.hand.GetTransform()[2,3] - robot.GetTransform()[2,3])
    def f_ee_extent(x):
        robot.SetActiveDOFValues(x)
        return w[2] * np.linalg.norm(robot.arm.hand.GetTransform()[:2,3] - robot.GetTransform()[:2,3])
    xyz = goal_T[:3,-1].tolist()
    wxyz = orpy.quatFromRotationMatrix(goal_T[:3,:3]).tolist()
    ik_solns = get_ik_solns(robot, goal_T)
    for i in range(min(20, ik_solns.shape[0])):
        init_joint_target = ik_solns[i].tolist()
        request = {
            'basic_info' : {
                'n_steps' : num_steps,
                'manip' : robot.arm.GetName(),
                'start_fixed' : True
            },
            'costs' : [
                {
                    'type' : 'joint_vel', # \sum_{t,j} (x_{t+1,j} - x_{t,j})^2
                    'params': {'coeffs' : [w[0]]}
                },
                {
                    'type' : 'collision',
                    'params' : {
                        'coeffs' : [20],
                        'dist_pen' : [0.025]
                    }
                }
            ],
            'constraints' : [
                {
                    'type' : 'pose',
                    'params' : {
                        'xyz': xyz,
                        'wxyz': wxyz,
                        'link': robot.arm.hand.GetName(),
                        'timestep': num_steps - 1
                    }
                }
            ],
            'init_info' : {
                'type' : 'straight_line',
                'endpoint' : init_joint_target
            }
        }
        s = json.dumps(request)
        prob = trajoptpy.ConstructProblem(s, env)
        for t in range(num_steps):
            prob.AddCost(f_ee_height, [(t,j) for j in xrange(7)], 'up%i'%t)
            prob.AddCost(f_ee_extent, [(t,j) for j in xrange(7)], 'up%i'%t)
        result = trajoptpy.OptimizeProblem(prob)
        waypoints = result.GetTraj()
        is_safe = traj_is_safe(
            waypoints, robot)
        if is_safe:
            return waypoints_to_traj(env, robot, waypoints)
    print('Failed to solve.')
