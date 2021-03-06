import argparse
import time
from mongoengine import connect
from ComparisonDocument import Comparison, binary_to_array
import utils
import numpy as np


def main(exp_name):
    connect('style_experiment')
    env, robot = utils.setup(render=False)
    while True:
        unlabeled = Comparison.objects(exp_name=exp_name, label=None)
        count = unlabeled.count()
        if count >= 30:
            successful = 0
            for c in unlabeled:
                wpsA = binary_to_array(c.wpsA)
                wpsB = binary_to_array(c.wpsB)
                with env:
                    true_costA = utils.ee_traj_cost(wpsA, robot)
                    true_costB = utils.ee_traj_cost(wpsB, robot)
                if np.abs(true_costA - true_costB) > .01:
                    c.label = (true_costB < true_costA)
                    successful += 1
                else:
                    c.label = -1
                c.save()
                print('Labeled {:d}/{:d} successfully.'.format(successful, count))
        time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    args = parser.parse_args()
    main(args.exp_name)
