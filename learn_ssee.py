from models import CostFunction
import planners
import utils
from utils import ee_traj_cost
import constants
import argparse
import numpy as np
import trajoptpy.math_utils as mu

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('learn_sum_squared_end_eff_disp')
ex.observers.append(MongoObserver.create())

@ex.config
def cfg():
    normalize = True
    dropout = 0.5
    activation = 'relu'
    iterations = 20
    save_path = './saves/learned_ssee0/'
    lr = .001
    batchnorm = False


@ex.automain
def main(iterations,
         activation,
         normalize,
         dropout,
         save_path,
         lr,
         batchnorm):
    env, robot = utils.setup(render=False)
    cf = CostFunction(
        robot,
        num_dofs=7,
        normalize=normalize,
        activation=activation,
        dropout=dropout, lr=lr,
        batchnorm=batchnorm)
    custom_cost = {'NN': planners.get_trajopt_error_cost(cf)}

    tqs = {}
    for idcs in constants.sg_pair_idcs[::2]:
        tqs[idcs] = utils.TrainingQueue(maxsize=1000)

    for idcs in constants.sg_pair_idcs[::2]:
        with env:
            linear = mu.linspace2d(
                constants.configs[idcs[0]],
                constants.configs[idcs[1]],
                10)
            for i in range(10):
                delta = utils.smooth_perturb(.1)
                wps_perturbed = linear + delta
                true_cost_perturbed = ee_traj_cost(wps_perturbed, robot)
                tqs[idcs].add((wps_perturbed, true_cost_perturbed))

    for i in range(iterations):
        print('[*] Iteration {:d}/{:d}'.format(i, iterations))
        # Training
        for j in range(100):
            for idcs in constants.sg_pair_idcs[::2]:
                tq = tqs[idcs]
                (wpsA, cA), (wpsB, cB) = tq.sample(num=2)
                if np.abs(cA - cB) < 0.01:
                    continue
                offset = np.random.uniform(0, 2*np.pi)
                wpsA[:,0] = offset
                wpsB[:,0] = offset
                cf.train_pref(wpsA[None], wpsB[None], [int(cB < cA)])

        random_inputs = np.random.uniform(0, 2*np.pi, size=(100, 10, 7))
        with env:
            labels = [ee_traj_cost(x, robot) for x in random_inputs]
        corr = cf.get_corrcoef(random_inputs, labels)
        ex.log_scalar('test.correlation', float(corr))

        # Generating training data
        for idcs in constants.sg_pair_idcs[::2]:
            with env:
                robot.SetActiveDOFValues(constants.configs[idcs[0]])
                wps = planners.trajopt_simple_plan(
                    env, robot, constants.configs[idcs[1]],
                    custom_traj_costs=custom_cost,
                    joint_vel_coeff=0).GetTraj()
                delta = utils.smooth_perturb(.1)
                wps_perturbed = wps + delta
                true_cost = ee_traj_cost(wps, robot)
                true_cost_perturbed = ee_traj_cost(wps_perturbed, robot)
            tqs[idcs].add((wps, true_cost))
            tqs[idcs].add((wps_perturbed, true_cost_perturbed))
    cf.save_model(save_path)
