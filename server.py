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

app = Flask(__name__)

display_env, display_robot1, display_robot2 = utils.setup(two_robots=True)

env, robot = utils.setup(render=False)
cf = CostFunction(robot, num_dofs=21, use_total_duration=True)
custom_cost = {'NN': planners.get_trajopt_cost(cf)}

with open('./data/world_space_trajs.pkl', 'rb') as f:
    starter_data = pickle.load(f)
    with env:
        wf = np.stack([utils.world_space_featurizer(robot, wps) for wps in starter_data])
    starter_data = np.concatenate([starter_data, wf], axis=-1)

with env:
    default_wps = planners.trajopt_simple_plan(env, robot, c.configs[1])
    default_traj = utils.waypoints_to_traj(
        display_env,
        display_robot1,
        default_wps.GetTraj(),
        10,
        None)

# data_q = utils.TrainingQueue(maxsize=100)
# for i in range(starter_data.shape[0] / 2):
#     data_q.add(starter_data[i])
training_q = utils.TrainingQueue(maxsize=2000)

to_label = []
# utils.add_samples_to_label(robot, to_label, data_q, num=10)


@app.route('/')
def index():
    return render_template('index.html',
                           num_to_label=len(to_label),
                           num_train_pts=len(training_q))


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
    label = None
    if traj_choice == 'A':
        label = 0
    elif traj_choice == 'B':
        label = 1
    xA, _, xB, _ = to_label.pop(0)
    if label is not None:
        data = ((xA, 10), (xB, 10), label)
        training_q.add(data)
    return redirect(url_for('index'))


@app.route('/generate', methods=['POST'])
def generate_trajs():
    num_trajs = int(request.form['num_trajs'])
    robot.SetActiveDOFValues(c.configs[0])
    with env:
        results = planners.trajopt_multi_plan(
            env,
            robot,
            c.configs[1],
            custom_traj_costs=custom_cost,
            num_inits=num_trajs)
        new_data = []
        for res in results:
            wps = res.GetTraj()
            wf = utils.world_space_featurizer(robot, wps)
            new_data.append(np.concatenate([wps, wf], axis=1))
        temp_to_label = []
        for xA, xB in itertools.combinations(new_data, 2):
            if not np.allclose(xA[:,:7], xB[:,:7]):
                trajA = utils.waypoints_to_traj(display_env, display_robot1, xA[:,:7], 10, None)
                trajB = utils.waypoints_to_traj(display_env, display_robot2, xB[:,:7], 10, None)
                temp_to_label.append((xA, trajA, xB, trajB))
        random.shuffle(temp_to_label)
        to_label.extend(temp_to_label)
    return redirect(url_for('index'))


@app.route('/train', methods=['POST'])
def handle_train():
    utils.train(cf, training_q, epochs=20)
    return redirect(url_for('index'))
