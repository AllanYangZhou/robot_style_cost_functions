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


featurizers = [feature_height, feature_orientation]


def compute_feature_values(robot, waypoints):
    result = {}
    for f in featurizers:
        total = 0
        for i in range(waypoints.shape[0]):
            old = robot.GetActiveDOFValues()
            total += f(robot, waypoints[i])
            robot.SetActiveDOFValues(old)
        mean = total / waypoints.shape[0]
        result[f.__name__] = mean
    return result


def trajopt_plan_to_config(env, robot, goal_config,
                           num_steps=10, w=[0, 0],
                           waypoints=[], duration=15,
                           retimer=None):
    def cost_height(x):
        val = w[0] * feature_height(robot, x)
        return val
    def cost_orientation(x):
        val = w[1] * feature_orientation(robot, x)
        return val
    def ee_timing_cost(waypoints):
        waypoints = waypoints.reshape((7, -1)).T
        ee_positions = np.stack([get_ee_coords(robot, x)
                                 for x in waypoints])
        ee_vel = np.square(np.diff(ee_positions, n=2, axis=0))
        return np.sum(ee_vel)
    start_joints = robot.GetActiveDOFValues()
    inits = []
    inits.append(mu.linspace2d(start_joints, goal_config, num_steps))
    waypoint_step = (num_steps - 1) // 2
    for waypoint in waypoints:
        inits.append(interpolate_waypoint(start_joints, waypoint, goal_config, num_steps))
    joint_target = goal_config.tolist()
    traj = None
    best_cost = np.inf
    combined_feature_values = []
    init_unchanged_alert = None
    for i, init in enumerate(inits):
        request = {
            'basic_info' : {
                'n_steps' : num_steps,
                'manip' : robot.arm.GetName(),
                'start_fixed' : True
            },
            'costs' : [
                {
                    'type' : 'joint_vel', # \sum_{t,j} (x_{t+1,j} - x_{t,j})^2
                    'params': {'coeffs' : [1]}
                },
                {
                    'type' : 'collision',
                    'params' : {
                        'coeffs' : [10 * 2],
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
        request['init_info']['data'] = init.tolist()
        s = json.dumps(request)
        prob = trajoptpy.ConstructProblem(s, env)
        for t in range(num_steps):
            prob.AddCost(
                cost_height,
                [(t,j) for j in xrange(7)],
                'height%i'%t)
            prob.AddCost(
                cost_orientation,
                [(t,j) for j in xrange(7)],
                'orientation%i'%t)
        # prob.AddCost(
        #     ee_timing_cost,
        #     [(t, j) for t in range(num_steps) for j in range(7)],
        #     'ee_timing')
        result = trajoptpy.OptimizeProblem(prob)
        waypoints = result.GetTraj()
        is_safe = traj_is_safe(
            waypoints, robot)
        unzipped = zip(*result.GetCosts())
        total_cost = np.sum(unzipped[1])
        cost_names = list(unzipped[0])
        collision_cost = sum([unzipped[1][k] for k in range(len(cost_names))
                              if 'collision' in cost_names[k]])
        smoothness_cost = unzipped[1][cost_names.index('joint_vel')]
        feature_values = compute_feature_values(robot, waypoints)
        feature_values['smoothness'] = smoothness_cost
        feature_values['collision'] = collision_cost
        feature_values['total'] = total_cost
        combined_feature_values.append(feature_values)
        if total_cost < best_cost:
            traj = waypoints_to_traj(env, robot, waypoints, duration, retimer)
            best_cost = total_cost
            if np.allclose(waypoints, init):
                init_unchanged_alert = 'Init {:d}: Result is same as init.'.format(i)
            else:
                init_unchanged_alert = None
    if init_unchanged_alert:
        print(init_unchanged_alert)
    return traj, combined_feature_values
