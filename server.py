from flask import Flask, render_template, request, redirect, url_for

from models import CostFunction
import planners
import constants as c
import utils

import time
import pickle
import numpy as np

app = Flask(__name__)

env, robot = utils.setup()
cf = CostFunction(robot, num_dofs=21, use_total_duration=True)

with open('./data/world_space_trajs.pkl', 'rb') as f:
    starter_data = pickle.load(f)
    with env:
        wf = np.stack([utils.world_space_featurizer(robot, wps) for wps in starter_data])
    starter_data = np.concatenate([starter_data, wf], axis=-1)

with env:
    default_wps = planners.trajopt_simple_plan(env, robot, c.configs[1])
    default_traj = utils.waypoints_to_traj(env, robot, default_wps.GetTraj(), 10, None)

data_q = utils.TrainingQueue(maxsize=100)
for i in range(starter_data.shape[0] / 2):
    data_q.add(starter_data[i])
training_q = utils.TrainingQueue(maxsize=2000)

to_label = []
utils.add_samples_to_label(robot, to_label, data_q, num=10)

@app.route('/')
def index():
    return render_template('index.html', num_remaining=len(to_label))


@app.route('/save/<name>')
def save_cf(name):
    cf.save_model('./data/saves/{:s}'.format(name))


@app.route('/play/<option>')
def play_traj(option):
    if option == 'A':
        traj = to_label[0][1]
    else:
        traj = to_label[0][3]
    starting_dofs = traj.GetWaypoint(0)[:7]
    robot.SetActiveDOFValues(starting_dofs)
    robot.ExecuteTrajectory(traj)
    time.sleep(0.5)
    robot.SetActiveDOFValues(starting_dofs)
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
