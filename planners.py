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
    world_space_featurizer,
    random_init_maker
)
import trajoptpy
import trajoptpy.math_utils as mu
from trajoptpy.check_traj import traj_is_safe
import constants as c


def feature_height(robot, x):
    robot.SetActiveDOFValues(x)
    return np.square(robot.arm.hand.GetTransform()[2,3] - .222)


def feature_orientation(robot, x):
    robot.SetActiveDOFValues(x)
    ee_dir = -1 * robot.GetJoints()[6].GetAxis()
    upward_dir = np.array([0, 0, 1.])
    return np.dot(ee_dir, upward_dir)


def trajopt_simple_plan(env, robot, goal_config,
                        num_steps=10, custom_costs={},
                        init=None, custom_traj_costs={},
                        request_callbacks=[],
                        joint_vel_coeff=1.):
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
            },
            {
                'type' : 'joint_vel',
                'params': {'coeffs' : [joint_vel_coeff]}
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
    [callback(request) for callback in request_callbacks]
    s = json.dumps(request)
    prob = trajoptpy.ConstructProblem(s, env)
    for cost_name in custom_costs:
        cost_fn = custom_costs[cost_name]
        for t in range(num_steps):
            prob.AddCost(
                cost_fn,
                [(t,j) for j in range(7)],
                '{:s}{:d}'.format(cost_name, t))
    for cost_name in custom_traj_costs:
        cost_fn = custom_traj_costs[cost_name]
        if isinstance(cost_fn, tuple):
            prob.AddErrorCost(
                cost_fn[0],
                cost_fn[1],
                [(t,j) for t in range(num_steps) for j in range(7)],
                'SQUARED',
                cost_name)
        else:
            prob.AddCost(
                cost_fn,
                [(t,j) for t in range(num_steps) for j in range(7)],
                cost_name)
    result = trajoptpy.OptimizeProblem(prob)
    return result


def trajopt_multi_plan(env, robot, goal_config, num_inits=10, num_steps=10, warn_if_unchanged=False, **args):
    start_joints = robot.GetActiveDOFValues()
    linear_init = mu.linspace2d(start_joints, goal_config, num_steps)
    results = []
    default_res = trajopt_simple_plan(
        env,
        robot,
        goal_config,
        num_steps=num_steps,
        **args)
    results.append(default_res)
    for i in range(num_inits - 1):
        modified_init = random_init_maker(linear_init, one_wp=True)
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
        score = cf.get_traj_cost(x[None])
        return score
    return cf_cost


def get_trajopt_error_cost(cf):
    def cf_cost(x):
        x = x.reshape((10,7))
        score = cf.get_mlp_out(x)
        return score
    def cf_cost_grad(x):
        x = x.reshape((10, 7))
        return cf.get_grad_mlp_out(x)
    return cf_cost, cf_cost_grad
