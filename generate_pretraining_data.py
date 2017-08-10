import argparse
import pickle
import numpy as np
import openravepy as orpy
import utils
import planners
from planners import feature_orientation, feature_height


parser = argparse.ArgumentParser()
parser.add_argument('out_path', type=str)
parser.add_argument('--num_configs', type=int, default=100)
args = parser.parse_args()

env, robot = utils.setup(render=False)

w = [2, 0]
def cost_height(x):
    return w[0] * feature_height(robot, x)
def cost_orientation(x):
    return w[1] * feature_orientation(robot, x)
custom_costs = {
    'height': cost_height,
    'orientation': cost_orientation
}

start_T = np.array([[ 0.17016883, -0.77939717, -0.60297812,  0.40267832],
                    [-0.08240443,  0.59850436, -0.79687016, -0.01887856],
                    [ 0.98196338,  0.18529053,  0.03762101,  0.76589522],
                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
goal_T = np.array([[-0.74517836,  0.65465541,  0.12702559, -0.64859624],
                   [-0.66686438, -0.73123222, -0.14349685, -0.01553596],
                   [-0.00105579, -0.19163959,  0.9814648 ,  0.22241211],
                   [ 0.        ,  0.        ,  0.        ,  1.        ]])

start_configs = robot.arm.FindIKSolutions(start_T, orpy.IkFilterOptions.CheckEnvCollisions)
end_configs = robot.arm.FindIKSolutions(goal_T, orpy.IkFilterOptions.CheckEnvCollisions)
np.random.shuffle(start_configs)
np.random.shuffle(end_configs)
#cutoff = args.num_configs
start_configs = start_configs[:100]
end_configs = end_configs[:10]

count = 0
results = []
for start_config in start_configs:
    for end_config in end_configs:
        if count % 200 == 0:
            print('Reached iteration {:d}'.format(count))
        robot.SetActiveDOFValues(start_config)
        result = planners.trajopt_simple_plan(
            env,
            robot,
            end_config,
            custom_costs=custom_costs)
        results.append(result)
        count += 1

X_pretrain = []
y_pretrain = []
for result in results:
    X = result.GetTraj()
    y = utils.evaluate_costs(robot, X, custom_costs)
    augmented_X = utils.append_ee_position(robot, X)
    X_pretrain.append(augmented_X)
    X_pretrain.append(augmented_X[::-1])
    y_pretrain.append(y)
    y_pretrain.append(y)
X_pretrain = np.stack(X_pretrain)
y_pretrain = np.array(y_pretrain)
data = (X_pretrain, y_pretrain)

with open(args.out_path, 'wb') as f:
    pickle.dump(data, f)
    print('Completed.')
