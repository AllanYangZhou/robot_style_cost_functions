from flask import Flask, render_template, request, redirect, url_for

from models import CostFunction
import planners
import constants as c
import utils

import random
import time
import pickle
import itertools
import numpy as np
import trajoptpy.math_utils as mu
import pickle

app = Flask(__name__)

display_env, display_robot1, display_robot2 = utils.setup(two_robots=True)

env, robot = utils.setup(render=False)
opt_normalize = True
opt_recurrent = False
opt_total_duration = False
cf = CostFunction(
    robot,
    num_dofs=21,
    normalize=opt_normalize,
    recurrent=opt_recurrent,
    use_total_duration=opt_total_duration)
custom_cost = {'NN': planners.get_trajopt_cost(cf)}

with open('./data/world_space_trajs.pkl', 'rb') as f:
    starter_data = pickle.load(f)
    with env:
        wf = np.stack([utils.world_space_featurizer(robot, wps) for wps in starter_data])
    starter_data = np.concatenate([starter_data, wf], axis=-1)

training_q = utils.TrainingQueue(maxsize=2000)
to_label = []


@app.route('/')
def index():
    return render_template('index.html',
                           num_to_label=len(to_label),
                           num_train_pts=len(training_q),
                           opt_normalize=opt_normalize,
                           opt_recurrent=opt_recurrent,
                           opt_total_duration=opt_total_duration)


@app.route('/save/<name>')
def save_cf(name):
    cf.save_model('./data/saves/{:s}'.format(name))


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
        data = ((xA, 10), (xB, 10), label)
        training_q.add(data)
    if len(to_label):
        starting_dofs = to_label[0][1].GetWaypoint(0)[:7]
        display_robot1.SetActiveDOFValues(starting_dofs)
        display_robot2.SetActiveDOFValues(starting_dofs)
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
            new_data = []
            if not use_planning:
                wps = []
                linear_init = mu.linspace2d(start, goal, 10)
                wps.append(linear_init)
                for i in range(3):
                    modified_init = utils.random_init_maker(linear_init, one_wp=True)
                    wps.append(modified_init)
            else:
                results = planners.trajopt_multi_plan(
                    env,
                    robot,
                    goal,
                    custom_traj_costs=custom_cost,
                    num_inits=5)
                wps = [res.GetTraj() for res in results]
            for wp in wps:
                wf = utils.world_space_featurizer(robot, wp)
                new_data.append(np.concatenate([wp, wf], axis=1))
            for xA, xB in itertools.combinations(new_data, 2):
                if not np.allclose(xA[:,:7], xB[:,:7]):
                    trajA = utils.waypoints_to_traj(
                        display_env,
                        display_robot1,
                        xA[:,:7],
                        0.5,
                        None)
                    trajB = utils.waypoints_to_traj(
                        display_env,
                        display_robot2,
                        xB[:,:7],
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
    utils.train(cf, training_q, epochs=20)
    return redirect(url_for('index'))


@app.route('/save', methods=['POST'])
def handle_save():
    save_name = request.form['save_name']
    cf.save_model('./saves/' + save_name + '/')
    return redirect(url_for('index'))


@app.route('/load', methods=['POST'])
def handle_load():
    load_name = request.form['load_name']
    cf.load_model('./saves/' + load_name + '/')
    return redirect(url_for('index'))


@app.route('/save_tq', methods=['POST'])
def handle_save_tq():
    save_name = request.form['save_name']
    with open('./data/' + save_name, 'wb') as f:
        pickle.dump(training_q, f)
    return redirect(url_for('index'))


@app.route('/test_model')
def handle_test():
    robot.SetActiveDOFValues(c.configs[0])
    current = display_robot1.GetActiveDOFValues()
    display_robot1.SetActiveDOFValues(c.configs[0])
    display_robot2.SetActiveDOFValues(c.configs[0])
    with env:
        result = planners.trajopt_simple_plan(
            env,
            robot, c.configs[1],
            custom_traj_costs=custom_cost)
        default_result = planners.trajopt_simple_plan(
            env,
            robot, c.configs[1])
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
    return ''
