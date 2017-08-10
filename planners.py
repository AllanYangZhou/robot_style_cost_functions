import numpy as np
import openravepy as orpy
import json
from utils import (
    waypoints_to_traj,
    get_ik_solns,
    interpolate_waypoint,
    check_trajs_equal,
    get_ee_coords)
import trajoptpy
import trajoptpy.math_utils as mu
from trajoptpy.check_traj import traj_is_safe
import constants as c


def feature_height(robot, x):
    robot.SetActiveDOFValues(x)
    return np.square(robot.arm.hand.GetTransform()[2,3] - .222)


def feature_extent(robot, x):
    print('SHOULD NOT BE CALLED.')
    robot.SetActiveDOFValues(x)
    return np.linalg.norm(robot.arm.hand.GetTransform()[:2,3] -
                          robot.GetTransform()[:2,3])


def feature_orientation(robot, x):
    robot.SetActiveDOFValues(x)
    ee_dir = -1 * robot.GetJoints()[6].GetAxis()
    upward_dir = np.array([0, 0, 1.])
    return np.dot(ee_dir, upward_dir)


def trajopt_simple_plan(env, robot, goal_config,
                        num_steps=10, custom_costs={},
                        init=None, custom_traj_costs={}):
    start_joints = robot.GetActiveDOFValues()
    if init is None:
        init = mu.linspace2d(start_joints, goal_config, num_steps)
    request = {
        'basic_info' : {
            'n_steps' : num_steps,
            'manip' : robot.arm.GetName(),
            'start_fixed' : True
        },
        'costs' : [
            {
                'type' : 'joint_vel',
                'params': {'coeffs' : [1]}
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
                'params' : {'vals': goal_config.tolist()}
            }
        ],
        'init_info' : {
            'type' : 'given_traj',
            'data': init.tolist()
        }
    }
    s = json.dumps(request)
    prob = trajoptpy.ConstructProblem(s, env)
    for t in range(num_steps):
        for cost_name in custom_costs:
            cost_fn = custom_costs[cost_name]
            prob.AddCost(
                cost_fn,
                [(t,j) for j in range(7)],
                '{:s}{:d}'.format(cost_name, t))
    for cost_name in custom_traj_costs:
        cost_fn = custom_traj_costs[cost_name]
        prob.AddCost(
            cost_fn,
            [(t,j) for t in range(num_steps) for j in range(7)],
            cost_name)
    result = trajoptpy.OptimizeProblem(prob)
    return result


def trajopt_plan_to_config(env, robot, goal_config,
                           num_steps=10, w=[0, 0],
                           waypoints=[], duration=15,
                           retimer=None):
    def cost_height(x):
        return w[0] * feature_height(robot, x)
    def cost_orientation(x):
        return w[1] * feature_orientation(robot, x)
    custom_costs = {
        'height': cost_height,
        'orientation': cost_orientation
    }
    start_joints = robot.GetActiveDOFValues()
    inits = []
    inits.append(mu.linspace2d(start_joints, goal_config, num_steps))
    waypoint_step = (num_steps - 1) // 2
    for waypoint in waypoints:
        inits.append(interpolate_waypoint(start_joints, waypoint, goal_config, num_steps))
    joint_target = goal_config.tolist()
    traj = None
    best_cost = np.inf
    init_unchanged_alert = None
    for i, init in enumerate(inits):
        result = trajopt_simple_plan(
            env,
            robot,
            goal_config,
            num_steps=num_steps,
            init=init,
            custom_costs=custom_costs)
        waypoints = result.GetTraj()
        is_safe = traj_is_safe(
            waypoints, robot)
        total_cost = np.sum(zip(*result.GetCosts())[1])
        if total_cost < best_cost:
            traj = waypoints_to_traj(env, robot, waypoints, duration, retimer)
            best_cost = total_cost
            if np.allclose(waypoints, init):
                init_unchanged_alert = 'Init {:d}: Result is same as init.'.format(i)
            else:
                init_unchanged_alert = None
    if init_unchanged_alert:
        print(init_unchanged_alert)
    return traj
