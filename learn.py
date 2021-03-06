import multiprocessing
import Queue
import os
import shutil
import time
import psutil
import argparse

import numpy as np
import trajoptpy.math_utils as mu
from mongoengine import connect
import tensorflow as tf

from models import CostFunction
from linear_models import LinearCostFunction
import utils
import planners
import constants
import record_video
from ComparisonDocument import (
    Comparison,
    ComparisonQueue,
    array_to_binary,
    binary_to_array)
from tf_utils import add_simple_summary


label_batchsize = 30


def make_perturbs(x, num, perturb_amount):
    perturbed_trajs = []
    for j in range(num):
        delta = utils.smooth_perturb(perturb_amount)
        perturbed_trajs.append(x+delta)
    return perturbed_trajs


def comms_proc(exp_name, task_queue, traj_queue):
    proc = psutil.Process(os.getpid())

    connect('style_experiment') # mongodb connection

    traj_tqs = {idcs: utils.TrainingQueue(maxsize=20) for idcs in constants.sg_train_idcs}
    pairs_tracker = set()
    added_trajs = 0
    while True:
        try:
            # Clear out traj_queue
            while True:
                wps, path, idcs, counter= traj_queue.get(block=False)
                traj_tqs[idcs].add((wps, path, counter))
                added_trajs += 1
        except Queue.Empty:
            pass

        if len(Comparison.objects(exp_name=exp_name, label=None)) < label_batchsize and\
           added_trajs >= 10 * len(constants.sg_train_idcs):
            idx = np.random.choice(len(constants.sg_train_idcs))
            idcs = constants.sg_train_idcs[idx]
            traj_tq = traj_tqs[idcs]
            if len(traj_tq) < 10:
                continue
            (wpsA, pathA, counterA), (wpsB, pathB, counterB) = traj_tq.sample(num=2)
            pair_id = (min(counterA, counterB), max(counterA, counterB))
            if not np.allclose(wpsA, wpsB) and pair_id not in pairs_tracker:
                pairs_tracker.add(pair_id)
                c = Comparison(
                    exp_name=exp_name,
                    wpsA=array_to_binary(wpsA),
                    wpsB=array_to_binary(wpsB),
                    pathA=pathA, pathB=pathB)
                c.save()
        time.sleep(.1)


def main(args):
    exp_name = args.exp_name

    traj_counter = 0
    # Directory setup
    vid_dir = os.path.join('web', 'vids', exp_name)
    if os.path.exists(vid_dir):
        print('Video directory already exists!')
        traj_counter = len(os.listdir(vid_dir))
    else:
        os.makedirs(vid_dir)

    # Queue to send child process commands.
    task_queue = multiprocessing.Queue()
    # Queue for receiving trajs generated by child process.
    traj_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=comms_proc,
        args=[exp_name, task_queue, traj_queue]
    )
    p.start()

    client = connect('style_experiment')

    planned_wps = {idcs: None for idcs in constants.sg_train_idcs}
    # Training queue of *labeled* comparisons
    labeled_comparison_queue = ComparisonQueue(exp_name)

    env, robot = utils.setup(render=True)
    raw_input('Press enter to continue...')
    monitor = record_video.get_geometry()
    # cf = CostFunction(
    #     robot,
    #     use_all_links=True,
    #     quadratic=False)
    cf = LinearCostFunction(robot, num_dofs=50)
    cf_save_path = os.path.join('saves', 'style_experiment', exp_name, 'model')
    custom_cost = {'NN': planners.get_trajopt_cost(cf)}
    summary_writer = tf.summary.FileWriter(os.path.join('tb_logs', 'style_experiment', exp_name))

    # Actual main loop
    global_step = 0
    prev_num_labeled = 0
    num_perturbs = 0
    while True:
        num_labeled = len(labeled_comparison_queue)
        sufficient_labeled_data = num_labeled >= label_batchsize

        # Training step
        if num_labeled - prev_num_labeled > label_batchsize and num_perturbs > 5:
            prev_num_labeled = num_labeled
            num_perturbs = 0
            for _ in range(num_labeled):
                comp = labeled_comparison_queue.sample()
                wpsA = binary_to_array(comp.wpsA)
                wpsB = binary_to_array(comp.wpsB)
                label = comp.label
                wpsA_batch = np.repeat(wpsA[None], 8, axis=0)
                wpsB_batch = np.repeat(wpsB[None], 8, axis=0)
                label_batch = np.repeat(label, 8)
                for batch_idx in range(8):
                    offset = np.random.uniform(0, 2*np.pi)
                    wpsA_batch[batch_idx,:,0] = offset
                    wpsB_batch[batch_idx,:,0] = offset
                cf.train_pref(wpsA_batch, wpsB_batch, label_batch)
            wpsA_list, wpsB_list, labels = [], [], []
            for comp in labeled_comparison_queue.queue:
                wpsA_list.append(binary_to_array(comp.wpsA))
                wpsB_list.append(binary_to_array(comp.wpsB))
                labels.append(comp.label)
            wpsA_list, wpsB_list, labels = np.stack(wpsA_list), np.stack(wpsB_list), np.array(labels)
            pred = cf.get_traj_cost(wpsB_list) < cf.get_traj_cost(wpsA_list)
            t_acc = np.mean(labels == pred)
            add_simple_summary(summary_writer, 'training.accuracy', t_acc, global_step)
            summary_writer.flush()
            cf.save_model(cf_save_path, step=global_step)
            global_step += 1
            for idcs in planned_wps:
                planned_wps[idcs] = None

        # Generation step
        for idcs in constants.sg_train_idcs:
            s_idx, g_idx = idcs
            wps = planned_wps[idcs]
            if wps is None:
                q_s = constants.configs[s_idx]
                q_g = constants.configs[g_idx]
                with env:
                    robot.SetActiveDOFValues(q_s)
                    if sufficient_labeled_data:
                        wps = planners.trajopt_simple_plan(
                            env, robot, q_g,
                            custom_costs=custom_cost,
                            joint_vel_coeff=1).GetTraj()
                    else:
                        # Generate pretraining data
                        wps = mu.linspace2d(q_s, q_g, 10)
                planned_wps[idcs] = wps
            else:
                wps = make_perturbs(wps, 1, .1)[0]
                num_perturbs += 1
            vid_name = 's{:d}-g{:d}_traj{:d}.mp4'.format(
                s_idx, g_idx, traj_counter)
            out_path = os.path.join(vid_dir, vid_name)
            with env:
                traj = utils.waypoints_to_traj(env, robot, wps, 1, None)
            # TODO: move the video export to a different process
            if not args.no_record:
                record_video.record(robot, traj, out_path, monitor=monitor)
            traj_queue.put((wps, out_path, idcs, traj_counter))
            traj_counter += 1
        time.sleep(.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--no_record', action='store_true')
    args = parser.parse_args()

    main(args)
