from flask import Flask, render_template, request, redirect, url_for

from models import CostFunction
import planners
import constants as c
import utils

import random
import time
import itertools
import numpy as np
import trajoptpy.math_utils as mu
import pickle

app = Flask(__name__)

display_env, display_robot1, display_robot2 = utils.setup(two_robots=True)

env, robot = utils.setup(render=False)
cf = CostFunction(
    robot,
    num_dofs=7,
    normalize=True)
custom_cost = {'NN': planners.get_trajopt_error_cost(cf)}


to_label = []
session_vars = {
    'session_name': None,
    'human_batch_num': 0,
    'cost_train_num': 0,
    'tq': utils.TrainingQueue(maxsize=2000),
    'joint_vel_coeff': 1.
}


@app.route('/')
def index():
    return render_template('index.html',
                           num_to_label=len(to_label),
                           num_train_pts=len(session_vars['tq']),
                           session_vars=session_vars)


@app.route('/name_session', methods=['POST'])
def handle_name_session():
    name = request.form['session_name']
    make_or_load = request.form['make_or_load']
    session_vars['session_name'] = name
    if make_or_load == 'load':
        with open('./data/experiments/' + session_vars['session_name'] + '.pkl', 'rb') as f:
            old_sess = pickle.load(f)
        for key in old_sess:
            session_vars[key] = old_sess[key]
        cf.load_model('./saves/experiments/' + session_vars['session_name'] + '/')
    return redirect(url_for('index'))


@app.route('/play')
def play_traj():
    trajA = to_label[0][1]
    trajB = to_label[0][3]
    starting_dofs = trajA.GetWaypoint(0)[:7]

    display_robot1.SetActiveDOFValues(starting_dofs)
    display_robot2.SetActiveDOFValues(starting_dofs)

    display_robot1.GetController().SetPath(trajA)
    display_robot2.GetController().SetPath(trajB)

    display_robot1.WaitForController(0)
    display_robot2.WaitForController(0)

    time.sleep(0.5)
    display_robot1.SetActiveDOFValues(starting_dofs)
    display_robot2.SetActiveDOFValues(starting_dofs)

    return ''


@app.route('/submit', methods=['POST'])
def handle_submission():
    traj_choice = request.form['traj_choice']
    if traj_choice == 'skip':
        to_label[:] = []
        return redirect(url_for('index'))

    label = None
    if traj_choice == 'A':
        label = 0
    elif traj_choice == 'B':
        label = 1
    xA, _, xB, _ = to_label.pop(0)
    if label is not None:
        data = (xA, xB, label)
        session_vars['tq'].add(data)
    if len(to_label):
        starting_dofs = to_label[0][1].GetWaypoint(0)[:7]
        display_robot1.SetActiveDOFValues(starting_dofs)
        display_robot2.SetActiveDOFValues(starting_dofs)
    else:
        session_vars['human_batch_num'] += 1
    return redirect(url_for('index'))


@app.route('/generate', methods=['POST'])
def generate_trajs():
    use_planning = (request.form['generate_option'] == 'planning')
    temp_to_label = []
    with env:
        sg_pair_idcs = np.random.choice(range(len(c.start_goal_pairs)), size=4, replace=False)
        for idx in sg_pair_idcs:
            start, goal = c.start_goal_pairs[idx]
            robot.SetActiveDOFValues(start)
            if not use_planning:
                given = mu.linspace2d(start, goal, 10)
            else:
                given = planners.trajopt_simple_plan(
                    env,
                    robot,
                    goal,
                    custom_traj_costs=custom_cost,
                    joint_vel_coeff=session_vars['joint_vel_coeff']).GetTraj()
            wps = [given]
            for i in range(3):
                delta = utils.smooth_perturb(.1)
                wps.append(given + delta)
            for xA, xB in itertools.combinations(wps, 2):
                if not np.allclose(xA[:,:7], xB[:,:7]):
                    trajA = utils.waypoints_to_traj(
                        display_env,
                        display_robot1,
                        xA,
                        0.5,
                        None)
                    trajB = utils.waypoints_to_traj(
                        display_env,
                        display_robot2,
                        xB,
                        0.5,
                        None)
                    temp_to_label.append((xA, trajA, xB, trajB))
    random.shuffle(temp_to_label)
    to_label.extend(temp_to_label)
    starting_dofs = to_label[0][1].GetWaypoint(0)[:7]
    display_robot1.SetActiveDOFValues(starting_dofs)
    display_robot2.SetActiveDOFValues(starting_dofs)
    return redirect(url_for('index'))


@app.route('/train', methods=['POST'])
def handle_train():
    num_epochs = int(request.form['num_epochs'])
    utils.train(cf, session_vars['tq'], epochs=20)
    cf.save_model('./saves/experiments/' + session_vars['session_name'] + '/',
                  step=session_vars['cost_train_num'])
    session_vars['cost_train_num'] += 1
    return redirect(url_for('index'))


@app.route('/load', methods=['POST'])
def handle_load():
    load_name = request.form['load_name']
    cf.load_model('./saves/' + load_name + '/')
    return redirect(url_for('index'))


@app.route('/save_session', methods=['POST'])
def handle_save_session():
    with open('./data/experiments/' + session_vars['session_name'] + '.pkl', 'wb') as f:
        pickle.dump(session_vars, f)
    cf.save_model('./saves/experiments/' + session_vars['session_name'] + '/',
                  step=session_vars['cost_train_num'])
    return redirect(url_for('index'))


@app.route('/test_model', methods=['POST'])
def handle_test():
    test_num = int(request.form['test_num'])
    if test_num >= len(c.sg_test_idcs):
        return ''
    sidx, gidx = c.sg_test_idcs[test_num]
    start = c.configs[sidx]
    goal = c.configs[gidx]
    robot.SetActiveDOFValues(start)
    current = display_robot1.GetActiveDOFValues()
    display_robot1.SetActiveDOFValues(start)
    display_robot2.SetActiveDOFValues(start)
    with env:
        result = planners.trajopt_simple_plan(
            env,
            robot, goal,
            custom_traj_costs=custom_cost,
            joint_vel_coeff=session_vars['joint_vel_coeff'])
        default_result = planners.trajopt_simple_plan(
            env,
            robot, goal)
        traj = utils.waypoints_to_traj(display_env, display_robot1,
                                       result.GetTraj(), 0.5, None)
        default_traj = utils.waypoints_to_traj(display_env,
                                               display_robot2,
                                               default_result.GetTraj(),
                                               0.5, None)
    display_robot1.GetController().SetPath(traj)
    display_robot2.GetController().SetPath(default_traj)

    display_robot1.WaitForController(0)
    display_robot2.WaitForController(0)
    display_robot1.SetActiveDOFValues(current)
    display_robot2.SetActiveDOFValues(current)
    return redirect(url_for('index'))


@app.route('/set_joint_vel_coeff', methods=['POST'])
def handle_set_joint_vel_coeff():
    joint_vel_coeff = float(request.form['joint_vel'])
    session_vars['joint_vel_coeff'] = joint_vel_coeff
    return redirect(url_for('index'))
