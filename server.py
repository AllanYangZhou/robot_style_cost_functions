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

plot_dict = {'plots': None}

to_label = []
session_vars = {
    'session_name': None,
    'human_batch_num': 0,
    'cost_train_num': 0,
    'tq': utils.TrainingQueue(maxsize=2000),
    'joint_vel_coeff': 1.,
    'speed': 0.5,
    'use_ee_dist': False,
    'rotate_data': False,
    'loss_history': []
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
    if make_or_load == 'load':
        with open('./data/experiments/' + name + '.pkl', 'rb') as f:
            old_sess = pickle.load(f)
        for key in old_sess:
            session_vars[key] = old_sess[key]
        cf.load_model('./saves/experiments/' + name + '/')
    session_vars['session_name'] = name
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

@app.route('/plot')
def handle_plot():
    if plot_dict['plots'] is None:
        p1 = utils.plot_waypoints(display_env, display_robot1, to_label[0][0])
        p2 = utils.plot_waypoints(display_env, display_robot2, to_label[0][2])
        plot_dict['plots'] = (p1, p2)
    else:
        p1, p2 = plot_dict['plots']
        p1.Close()
        p2.Close()
        plot_dict['plots'] = None
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
        if session_vars['rotate_data']:
            offset = np.random.uniform(0, np.pi/4)
            for i in range(8):
                xA_rot = xA.copy()
                xB_rot = xB.copy()
                xA_rot[:,0] += i * (np.pi / 4) + offset
                xA_rot[:,0] += i * (np.pi / 4) + offset
                data = (xA_rot, xB_rot, label)
                session_vars['tq'].add(data)
        else:
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
                        session_vars['speed'],
                        None,
                        use_ee_dist=session_vars['use_ee_dist'])
                    trajB = utils.waypoints_to_traj(
                        display_env,
                        display_robot2,
                        xB,
                        session_vars['speed'],
                        None,
                        use_ee_dist=session_vars['use_ee_dist'])
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
    session_vars['loss_history'].append(
        utils.train(cf, session_vars['tq'], epochs=num_epochs))
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
                                       result.GetTraj(), session_vars['speed'], None,
                                       use_ee_dist=session_vars['use_ee_dist'])
        default_traj = utils.waypoints_to_traj(display_env,
                                               display_robot2,
                                               default_result.GetTraj(),
                                               0.5, None,
                                               use_ee_dist=session_vars['use_ee_dist'])
    display_robot1.GetController().SetPath(traj)
    display_robot2.GetController().SetPath(default_traj)

    display_robot1.WaitForController(0)
    display_robot2.WaitForController(0)
    display_robot1.SetActiveDOFValues(current)
    display_robot2.SetActiveDOFValues(current)
    return redirect(url_for('index'))


@app.route('/set_params', methods=['POST'])
def handle_set_joint_vel_coeff():
    joint_vel_coeff = float(request.form['joint_vel'])
    session_vars['speed'] = float(request.form['speed'])
    session_vars['joint_vel_coeff'] = joint_vel_coeff
    session_vars['use_ee_dist'] = (request.form['timing_dist'] == 'ee')
    session_vars['rotate_data'] = (request.form['rotate'] == 'yes')
    return redirect(url_for('index'))

