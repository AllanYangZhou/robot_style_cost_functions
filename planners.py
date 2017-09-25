import numpy as np
import openravepy as orpy
import json
from utils import (
    waypoints_to_traj,
    interpolate_waypoint,
    check_trajs_equal,
    get_ee_coords,
    get_ee_transform,
    normalize_vec,
    get_pos_ik_soln,
    world_space_featurizer)
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
                        init=None, custom_traj_costs={},
                        request_callbacks=[],
                        use_joint_vel=True):
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
    if use_joint_vel:
        request['costs'].append({
            'type' : 'joint_vel',
            'params': {'coeffs' : [1]}
        })
    [callback(request) for callback in request_callbacks]
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


def trajopt_multi_plan(env, robot, goal_config, num_inits=10, num_steps=10, warn_if_unchanged=False, **args):
    start_joints = robot.GetActiveDOFValues()
    linear_init = mu.linspace2d(start_joints, goal_config, num_steps)
    mid_idx = int(num_steps / 2)
    default_mid = linear_init[mid_idx]
    results = []
    default_res = trajopt_simple_plan(
        env,
        robot,
        goal_config,
        num_steps=num_steps,
        **args)
    results.append(default_res)
    for i in range(num_inits - 1):
        new_mid = np.random.multivariate_normal(default_mid, .2*np.eye(7))
        modified_init = np.concatenate([
            mu.linspace2d(linear_init[0], new_mid, mid_idx + 1),
            mu.linspace2d(new_mid, linear_init[-1], num_steps - mid_idx - 1)
        ], axis=0)
        res = trajopt_simple_plan(
            env,
            robot,
            goal_config,
            init=modified_init,
            num_steps=num_steps,
            **args)
        if warn_if_unchanged and np.allclose(modified_init, res.GetTraj()):
            print('Warning: unchanged init')
        if not np.allclose(default_res.GetTraj(), res.GetTraj()):
            results.append(res)
    return results


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


def collision_cost_callback(request):
    request['costs'][1]['params']['coeffs'] = [100]
    request['costs'][1]['params']['continuous'] = False


def make_pose_constraint_callback(pose_constraints):
    def callback(request):
        for t in pose_constraints:
            pose = orpy.poseFromMatrix(pose_constraints[t])
            quat, xyz = pose[:4], pose[4:]
            request['constraints'].append({
                'type': 'pose',
                'params': {
                    'timestep': t,
                    'xyz': xyz.tolist(),
                    'wxyz': quat.tolist(),
                    'link': 'j2s7s300_link_7'
                }
            })
    return callback


def modify_traj(env, robot, wps, num=1, verbose=False):
    new_results = []
    for i in range(num):
        if verbose and i % 10 == 0:
            print('Reached iteration {:d}.'.format(i))
        k=np.random.randint(4, 6)
        x0 = get_ee_coords(robot, wps[k-1])
        x1 = get_ee_coords(robot, wps[k])
        x2 = get_ee_coords(robot, wps[k+1])
        vert = np.array([0, 0, 1])
        n = normalize_vec(x2-x0)
        proj_vert_n = np.dot(vert, n) * n
        near_vert = normalize_vec(vert - proj_vert_n)
        horiz = normalize_vec(np.cross(n, near_vert))
        v = .0001
        U = np.stack([near_vert, horiz, n], axis=1)
        L = np.diag([v, v, 0])
        S = U.dot(L).dot(U.T)

        sample_pos = np.random.multivariate_normal(x1, S)
        soln = get_pos_ik_soln(robot, sample_pos)
        ee_T = get_ee_transform(robot, soln)
        rc = [
            collision_cost_callback,
            make_pose_constraint_callback({k: ee_T})]
        new_results.append(trajopt_simple_plan(
            env,
            robot,
            c.configs[1],
            request_callbacks=rc))
    return new_results


def get_trajopt_cost(cf):
    def cf_cost(x):
        x = x.reshape((10,7))
        f = world_space_featurizer(cf.robot, x)
        #f = np.concatenate([np.ones((10,1)), f], axis=1)
        score = cf.cost_traj(f)
        return score
    return cf_cost
