import multiprocessing
import Queue
import os
import shutil
import time
import psutil

import numpy as np
import trajoptpy.math_utils as mu
from mongoengine import connect
import tensorflow as tf

from models import CostFunction
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


KILL_TASK = 'kill'

num_perturbs = 10
label_batchsize = 30


def make_perturbs(x, num, perturb_amount):
    perturbed_trajs = [x]
    for j in range(num):
        delta = utils.smooth_perturb(perturb_amount)
        perturbed_trajs.append(x+delta)
    return perturbed_trajs


def comms_proc(task_queue, traj_queue):
    proc = psutil.Process(os.getpid())
    # Directory setup
    shutil.rmtree('web/vids/')
    os.mkdir('web/vids/')

    connect('style_experiment') # mongodb connection

    traj_tqs = {idcs: utils.TrainingQueue(maxsize=15) for idcs in constants.sg_train_idcs}
    new_trajs = 0
    # pairs_tracker = set()
    while True:
        try:
            next_task = task_queue.get(block=False)
            if next_task == KILL_TASK:
                print('Child process terminating.')
                break
        except Queue.Empty:
            pass

        try:
            # Clear out traj_queue
            while True:
                wps, path, idcs, counter= traj_queue.get(block=False)
                traj_tqs[idcs].add((wps, path, counter))
                new_trajs += 1
        except Queue.Empty:
            pass

        if len(Comparison.objects(label=None)) == 0 and new_trajs >= (len(constants.sg_train_idcs) * (num_perturbs + 1)):
            new_trajs = 0
            # only incremented if a comparison was successfully added.
            successes_count = 0
            for _ in range(100):
                if successes_count >= label_batchsize:
                    break
                idx = np.random.choice(len(constants.sg_train_idcs))
                idcs = constants.sg_train_idcs[idx]
                traj_tq = traj_tqs[idcs]
                (wpsA, pathA, counterA), (wpsB, pathB, counterB) = traj_tq.sample(num=2)
                pair_id = (min(counterA, counterB), max(counterA, counterB))
                if not np.allclose(wpsA, wpsB): # and pair_id not in pairs_tracker:
                    # pairs_tracker.add(pair_id)
                    c = Comparison(
                        wpsA=array_to_binary(wpsA),
                        wpsB=array_to_binary(wpsB),
                        pathA=pathA, pathB=pathB)
                    c.save()
                    successes_count += 1
        time.sleep(.1)


def main():
    client = connect('style_experiment')
    if 'style_experiment' in client.database_names():
        resp = raw_input('DB exists already. Clear?')
        if resp == 'y':
            client.drop_database('style_experiment')
        else:
            print('OK, exiting.')
            return

    # Queue to send child process commands.
    task_queue = multiprocessing.Queue()
    # Queue for receiving trajs generated by child process.
    traj_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=comms_proc,
        args=[task_queue, traj_queue]
    )
    p.start()

    # Training queue of *labeled* comparisons
    labeled_comparison_queue = ComparisonQueue()

    env, robot = utils.setup(render=True)
    raw_input('Press enter to continue...')
    monitor = record_video.get_geometry()
    cf = CostFunction(
        robot,
        use_all_links=False,
        quadratic=False)
    custom_cost = {'NN': planners.get_trajopt_cost(cf)}
    summary_writer = tf.summary.FileWriter(os.path.join('tb_logs', 'style_experiment'))

    # Actual main loop
    traj_counter = 0
    global_step = 0
    while True:
        sufficient_labeled_data = len(labeled_comparison_queue) >= label_batchsize

        # Training step
        if sufficient_labeled_data:
            qsize = len(labeled_comparison_queue)
            for _ in range(qsize):
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
            global_step += 1

        # Generation step
        for idcs in constants.sg_train_idcs:
            s_idx, g_idx = idcs
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
                perturbed_wps = make_perturbs(wps, 10, .1)
            for wps in perturbed_wps:
                out_path = 'web/vids/s{:d}-g{:d}_traj{:d}.webm'.format(
                    s_idx, g_idx, traj_counter)
                with env:
                    traj = utils.waypoints_to_traj(env, robot, wps, 1, None)
                # TODO: move the video export to a different process
                #record_video.record(robot, traj, out_path, monitor=monitor)
                traj_queue.put((wps, out_path, idcs, traj_counter))
                traj_counter += 1
        time.sleep(.1)

    # Tell the child process to end, then join.
    task_queue.put(KILL_TASK)
    p.join()


if __name__ == '__main__':
    main()
