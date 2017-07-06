import numpy as np
import openravepy as orpy
import json
from utils import waypoints_to_traj
import trajoptpy
import trajoptpy.math_utils as mu
from trajoptpy.check_traj import traj_is_safe, get_ik_solns


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
    best_cost = np.inf
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
        request['init_info']['data'] = init.tolist()
        s = json.dumps(request)
        prob = trajoptpy.ConstructProblem(s, env)
        for t in range(num_steps):
            prob.AddCost(
                f_ee_height,
                [(t,j) for j in xrange(7)],
                'up%i'%t)
            prob.AddCost(
                f_ee_extent,
                [(t,j) for j in xrange(7)],
                'up%i'%t)
        result = trajoptpy.OptimizeProblem(prob)
        waypoints = result.GetTraj()
        is_safe = traj_is_safe(
            waypoints, robot)
        total_cost = np.sum(zip(*result.GetCosts())[1])
        if is_safe and total_cost < best_cost:
            traj = waypoints_to_traj(env, robot, waypoints)
            best_cost = total_cost
    return traj
