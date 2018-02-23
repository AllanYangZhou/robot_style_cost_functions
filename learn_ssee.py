from models import CostFunction
import planners
import utils
from utils import ee_traj_cost
from tf_utils import add_simple_summary
import constants
import argparse
import numpy as np
import trajoptpy.math_utils as mu
import tensorflow as tf
import os

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.stflow import LogFileWriter

ex_name = 'learn_sum_squared_end_eff_disp'
ex = Experiment(ex_name)
ex.observers.append(MongoObserver.create())

@ex.config
def cfg():
    normalize = True
    dropout = 0.5
    activation = 'relu'
    iterations = 20
    lr = .001
    batchnorm = False
    q_length = 10000
    use_quadratic_features = True
    perturb_amount = .1
    provide_vel = True
    include_configs = False
    use_all_links = True
    h_size = 64
    random_inits = False


def make_perturbs(x, num, perturb_amount):
    perturbed_trajs = []
    for j in range(num):
        delta = utils.smooth_perturb(
            np.random.uniform(.01, 1) if perturb_amount == 'rand' else perturb_amount)
        perturbed_trajs.append(x+delta)
    return perturbed_trajs


@ex.automain
@LogFileWriter(ex) # Records location of logs
def main(iterations,
         activation,
         normalize,
         dropout,
         lr,
         batchnorm,
         q_length,
         use_quadratic_features,
         perturb_amount,
         provide_vel,
         include_configs,
         use_all_links,
         h_size,
         random_inits,
         _seed, _run):
    tf.set_random_seed(_seed) # Set tf's seed to exp seed
    env, robot = utils.setup(render=False)
    cf = CostFunction(
        robot,
        num_dofs=7,
        normalize=normalize,
        include_configs=include_configs,
        provide_vel=provide_vel,
        use_all_links=use_all_links,
        activation=activation,
        dropout=dropout, lr=lr,
        batchnorm=batchnorm,
        quadratic=use_quadratic_features,
        h_size=h_size)
    ex.info['num_params'] = cf.num_params
    ex.info['mlp_input_shape'] = cf.mlp.model.input_shape

    summary_writer = tf.summary.FileWriter(os.path.join('tb_logs', ex_name, str(_run._id)))

    if use_quadratic_features:
        custom_cost = {'NN': planners.get_trajopt_error_cost(cf)}
    else:
        custom_cost = {'NN': planners.get_trajopt_cost(cf)}

    tqs = {}
    for idcs in constants.sg_train_idcs:
        tqs[idcs] = utils.TrainingQueue(maxsize=q_length)

    # The avg cost of train trajs generated using the true cost fxn.
    BEST_COST_SYNTH = 0.13875292480745718

    for idcs in constants.sg_train_idcs:
        with env:
            linear = mu.linspace2d(
                constants.configs[idcs[0]],
                constants.configs[idcs[1]],
                10)
            for wps_perturbed in make_perturbs(linear, 50, perturb_amount):
                true_cost_perturbed = ee_traj_cost(wps_perturbed, robot)
                tqs[idcs].add((wps_perturbed, true_cost_perturbed))

    for i in range(iterations):
        print('[*] Iteration {:d}/{:d}'.format(i, iterations))
        # Training
        for j in range(500):
            for idcs in constants.sg_train_idcs:
                tq = tqs[idcs]
                (wpsA, cA), (wpsB, cB) = tq.sample(num=2)
                if np.allclose(cA, cB):
                    continue
                offset = np.random.uniform(0, 2*np.pi)
                wpsA[:,0] = offset
                wpsB[:,0] = offset
                cf.train_pref(wpsA[None], wpsB[None], [int(cB < cA)])

        random_inputs = np.random.uniform(0, 2*np.pi, size=(100, 10, 7))
        with env:
            labels = [ee_traj_cost(x, robot) for x in random_inputs]
        corr = cf.get_corrcoef(random_inputs, labels)
        add_simple_summary(summary_writer, 'test.correlation', corr, i)
        summary_writer.flush()

        # Generating training data
        true_cost_list = [] # want to keep track of how good our true costs are
        true_cost_variances = [] # Only applies if random_init = True
        for idcs in constants.sg_train_idcs:
            with env:
                robot.SetActiveDOFValues(constants.configs[idcs[0]])
                if random_inits:
                    results_list = planners.trajopt_multi_plan(
                        env, robot, constants.configs[idcs[1]], num_inits=10,
                        custom_traj_costs=(custom_cost if use_quadratic_features else {}),
                        custom_costs=({} if use_quadratic_features else custom_cost),
                        joint_vel_coeff=0)
                    true_costs = []
                    for res in results_list:
                        wps = res.GetTraj()
                        true_cost = ee_traj_cost(wps, robot)
                        tqs[idcs].add((wps, true_cost))
                        true_costs.append(true_cost)
                    true_cost_list.append(np.min(true_costs))
                    true_cost_variances.append(np.var(true_costs))
                else:
                    wps = planners.trajopt_simple_plan(
                        env, robot, constants.configs[idcs[1]],
                        custom_traj_costs=(custom_cost if use_quadratic_features else {}),
                        custom_costs=({} if use_quadratic_features else custom_cost),
                        joint_vel_coeff=0).GetTraj()
                    true_cost = ee_traj_cost(wps, robot)
                    tqs[idcs].add((wps, true_cost))
                    true_cost_list.append(true_cost)
                    for wps_perturbed in make_perturbs(x, 10, perturb_amount):
                        true_cost_perturbed = ee_traj_cost(wps_perturbed, robot)
                        tqs[idcs].add((wps_perturbed, true_cost_perturbed))
        add_simple_summary(summary_writer, 'training.true_cost', np.mean(true_cost_list), i)
        if random_inits:
            add_simple_summary(
                summary_writer,
                'training.true_cost_var',
                np.var(true_cost_variances, i))
        summary_writer.flush() # flush summary writer at end of iteration
    save_folder = './saves/learned_ssee/'
    cf.save_model(os.path.join(save_folder, 'model'))
    for fname in os.listdir(save_folder):
        ex.add_artifact(os.path.join(save_folder, fname))
