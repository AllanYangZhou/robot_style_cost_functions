import multiprocessing
import Queue
import os
import shutil
import time

import numpy as np
import trajoptpy.math_utils as mu
from mongoengine import connect

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


KILL_TASK = 'kill'

num_perturbs = 5
label_batchsize = 30


def make_perturbs(x, num, perturb_amount):
    perturbed_trajs = [x]
    for j in range(num):
        delta = utils.smooth_perturb(perturb_amount)
        perturbed_trajs.append(x+delta)
    return perturbed_trajs


def comms_proc(task_queue, traj_queue):
    # Directory setup
    shutil.rmtree('web/vids/')
    os.mkdir('web/vids/')

    connect('style_experiment') # mongodb connection

    traj_tqs = {idcs: utils.TrainingQueue(maxsize=20) for idcs in constants.sg_train_idcs}

    pretrain_trajs_generated = False
    pairs_tracker = set()
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
        except Queue.Empty:
            pass

        if not pretrain_trajs_generated:
            pretrain_trajs_generated = True
            for idcs in constants.sg_train_idcs:
                if len(traj_tqs[idcs]) < (num_perturbs + 1):
                    pretrain_trajs_generated = False
            # Don't start making pairs until the pretrain traj generation is done.
            continue

        if len(Comparison.objects(label=None)) == 0:
            attempts_count = 0
            # only incremented if a comparison was successfully added.
            successes_count = 0
            while successes_count < label_batchsize and attempts_count < 100:
                attempts_count += 1
                idx = np.random.choice(len(constants.sg_train_idcs))
                idcs = constants.sg_train_idcs[idx]
                traj_tq = traj_tqs[idcs]
                (wpsA, pathA, counterA), (wpsB, pathB, counterB) = traj_tq.sample(num=2)
                pair_id = (min(counterA, counterB), max(counterA, counterB))
                if not np.allclose(wpsA, wpsB) and pair_id not in pairs_tracker:
                    pairs_tracker.add(pair_id)
                    c = Comparison(
                        wpsA=array_to_binary(wpsA),
                        wpsB=array_to_binary(wpsB),
                        pathA=pathA, pathB=pathB)
                    c.save()
                    successes_count += 1
        time.sleep(.1)


def main():
    # Queue to send child process commands.
    task_queue = multiprocessing.Queue()
    # Queue for receiving trajs generated by child process.
    traj_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=comms_proc,
        args=[task_queue, traj_queue]
    )
    p.start()

    connect('style_experiment')
    labeled_comparison_queue = ComparisonQueue()

    env, robot = utils.setup(render=True)
    raw_input('Press enter to continue...')
    monitor = record_video.get_geometry()
    cf = CostFunction(
        robot,
        use_all_links=False,
        quadratic=False)
    custom_cost = {'NN': planners.get_trajopt_cost(cf)}
    # Training queue of *labeled* comparisons
    traj_counter = 0

    # Actual main loop
    made_pretrain_data =  False
    while True:
        if made_pretrain_data and len(labeled_comparison_queue) < label_batchsize:
            # We've generated the pretraining data but there
            # aren't enough labels yet to do anything else.
            time.sleep(.1)
            continue

        # Training step
        print('Training time')
        qsize = len(labeled_comparison_queue)
        for _ in range(qsize):
            comp = labeled_comparison_queue.sample()
            wpsA = binary_to_array(comp.wpsA)
            wpsB = binary_to_array(comp.wpsB)
            label = comp.label
            cf.train_pref(wpsA[None], wpsB[None], [label])

        # Generation step
        for idcs in constants.sg_train_idcs:
            s_idx, g_idx = idcs
            q_s = constants.configs[s_idx]
            q_g = constants.configs[g_idx]
            with env:
                robot.SetActiveDOFValues(q_s)
                if made_pretrain_data:
                    wps = planners.trajopt_simple_plan(
                        env, robot, q_g,
                        custom_costs=custom_cost,
                        joint_vel_coeff=1).GetTraj()
                else:
                    # Generate pretraining data
                    wps = mu.linspace2d(q_s, q_g, 10)
                    made_pretrain_data = True
                perturbed_wps = make_perturbs(wps, 5, .1)
            for wps in perturbed_wps:
                out_path = 'web/vids/s{:d}-g{:d}_traj{:d}.webm'.format(
                    s_idx, g_idx, traj_counter)
                with env:
                    traj = utils.waypoints_to_traj(env, robot, wps, 1, None)
                # TODO: move the video export to a different process
                record_video.record(robot, traj, out_path, monitor=monitor)
                traj_queue.put((wps, out_path, idcs, traj_counter))
                traj_counter += 1
        time.sleep(.1)

    # Tell the child process to end, then join.
    task_queue.put(KILL_TASK)
    p.join()


if __name__ == '__main__':
    main()
