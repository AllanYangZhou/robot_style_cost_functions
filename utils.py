import trajoptpy
from trajoptpy.check_traj import traj_is_safe
import trajoptpy.math_utils as mu
import interactpy
import openravepy as orpy
import numpy as np
import json
import constants as c


def setup():
    env, robot = interactpy.initialize()
    robot.SetActiveDOFs(robot.arm.GetArmIndices())
    robot.SetActiveDOFValues(c.starting_angles)
    robot.SetDOFValues(c.starting_finger_angles, [7,8,9])
    table = env.GetKinBody('table')
    mug = env.GetKinBody('mug')
    env.Remove(table)
    env.Remove(mug)
    env.Load('{:s}/table.xml'.format(c.iact_ctrl_path))
    env.Load('{:s}/cabinet.xml'.format(c.iact_ctrl_path))
    table = env.GetKinBody('table')
    cabinet = env.GetKinBody('cabinet')
    cabinet.SetTransform(c.cabinet_T)
    table.SetTransform(c.table_T)
    cabinet.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(c.cabinet_color)
    table.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(c.table_color)
    env.Load('/usr/local/share/openrave-0.9/data/box1.kinbody.xml')
    box = env.GetKinBody('box1')
    box.SetTransform(c.box_T)
    robot.Grab(box)
    viewer = env.GetViewer()
    viewer.SetCamera(c.camera_T)
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


def trajopt_plan_to_config(env, robot, goal_config,
                           num_steps=10, w=[1, 0, 0],
                           return_result=False,
                           waypoints=[]):
    def f_ee_height(x):
        robot.SetActiveDOFValues(x)
        return np.array([w[1] * (robot.arm.hand.GetTransform()[2,3] - robot.GetTransform()[2,3])])
    def f_ee_extent(x):
        robot.SetActiveDOFValues(x)
        return w[2] * np.linalg.norm(
            robot.arm.hand.GetTransform()[:2,3] - robot.GetTransform()[:2,3])
    start_joints = robot.GetActiveDOFValues()
    inits = []
    inits.append(mu.linspace2d(start_joints, goal_config, num_steps))
    waypoint_step = (num_steps - 1) // 2
    for waypoint in waypoints:
        init = np.empty((num_steps, 7))
        init[:waypoint_step+1] = mu.linspace2d(start_joints, waypoint, waypoint_step+1)
        init[waypoint_step:] = mu.linspace2d(waypoint, goal_config, num_steps - waypoint_step)
        inits.append(init)
    joint_target = goal_config.tolist()
    traj = None
    for init in inits:
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
                    'type' : 'joint',
                    'params' : {'vals': joint_target}
                }
            ],
            'init_info' : {
                'type' : 'given_traj'
            }
        }
        request['init_info']['data'] = [row.tolist() for row in init]
        s = json.dumps(request)
        prob = trajoptpy.ConstructProblem(s, env)
        for t in range(num_steps):
            prob.AddCost(
                f_ee_height,
                [(t,j) for j in xrange(7)],
                'up%i'%t)
            prob.AddCost(f_ee_extent, [(t,j) for j in xrange(7)], 'up%i'%t)
        result = trajoptpy.OptimizeProblem(prob)
        waypoints = result.GetTraj()
        is_safe = traj_is_safe(
            waypoints, robot)
        if is_safe:
            traj = waypoints_to_traj(env, robot, waypoints)
    if traj:
        if return_result:
            return traj, result
        else:
            return traj
    else:
        print('Failed to solve.')
