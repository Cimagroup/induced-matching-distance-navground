###################################
###################################
# EXPERIMENTS: PART 2
# CREATING THE SIGNALS
###################################
###################################

###################################
# 1: IMPORTING MODULES
###################################

import numpy as np
import matplotlib.pyplot as plt
import argparse
from functools import partial
from tslearn import metrics
from tqdm import tqdm
import gudhi
from gudhi.wasserstein import wasserstein_distance
import perdiver.perdiver as perdiver

poses = np.load('files/poses.npy')
types = np.load('files/types.npy')

###################################
# 2: ADJUSTING PARAMETERS
###################################

parser = argparse.ArgumentParser(description='Simulation Parameters')
parser.add_argument('--length', type=float, default=10.0, help='Length of the environment')
parser.add_argument('--width', type=float, default=10.0, help='Width of the environment')
parser.add_argument('--time_delay', type=int, default=1, help='Time delay to analise simulation intervals')
parser.add_argument('--embedding_length', type=int, default=10, help='Length of the simulation intervals')
parser.add_argument('--epsilon', type=int, default=1, help='Distance between intervals')

args = parser.parse_args([
        '--length', '15.0',
        '--width', '3.5',
        '--time_delay', '10',
        '--embedding_length', '6',
        '--epsilon', '50'
    ])

###################################
# 3: AUXILIAR FUNCTIONS
###################################

def normangle(angle):
    result = np.mod(angle, 2 * np.pi)
    result[result > np.pi] -= 2 * np.pi
    return result

def custom_distance(vector1, vector2, weights):
    result = 0
    if weights[0] != 0:
        px_diff = np.abs(vector1[0] - vector2[0])
        px_diff = np.minimum(px_diff, args.length - px_diff)
        result += px_diff * weights[0]
    if weights[1] != 0:
        py_diff = np.abs(vector1[1] - vector2[1])
        result += py_diff * weights[1]
    if weights[2] != 0:
        pr_diff = np.abs(vector1[2] - vector2[2])
        pr_diff = np.minimum(pr_diff, 2 * np.pi - pr_diff)
        result += pr_diff * weights[2]
    return result

weights = np.array([2/args.length,1/args.width,1/np.pi])
custom_distance_with_param = partial(custom_distance, weights=weights)

def dismat_from_steps(trajectories, steps):
    num_agents = trajectories.shape[1]
    dismat = np.zeros((num_agents, num_agents))
    for a in range(num_agents):
        for b in range(a+1):
            _, tsim = metrics.dtw_path_from_metric(trajectories[steps,a,:], trajectories[steps,b,:], metric=custom_distance_with_param)
            dismat[a,b] = tsim
    return dismat

def compute_dismat_list(trajectories, args):
    sim_steps = trajectories.shape[0]
    iterations = sim_steps - (args.embedding_length - 1) * args.time_delay
    dismat_list = []
    for i in range(iterations):
        steps = [i+args.time_delay*j for j in range(args.embedding_length)]
        dismat_list.append(dismat_from_steps(trajectories, steps))
    return dismat_list

def matching_distance(matching):
    return sum([abs(bar[0]-bar[1]) for bar in matching])

def compute_matching_signal(dismat_list):
    matching_signal = [matching_distance(perdiver.get_matching_diagram(dismat_list[i], dismat_list[i+args.epsilon]))
                                  for i in range(len(dismat_list)-args.epsilon)]
    return np.array(matching_signal)

def compute_statistics(data):
    median = np.median(data, axis=0)
    percentile_25 = np.percentile(data, 25, axis=0)
    percentile_75 = np.percentile(data, 75, axis=0)
    return median, percentile_25, percentile_75

###################################
# 4: CREATING THE SIGNALS
###################################

num_simulations = poses.shape[0]
matching_signals = np.empty((0,0))
for i in tqdm(range(num_simulations), desc="Progress: ", leave = False):
    trajectories = poses[i].copy()
    trajectories[:,:,2] = normangle(trajectories[:,:,2])
    dismat_list = compute_dismat_list(trajectories, args)
    signal = np.array(compute_matching_signal(dismat_list))
    if matching_signals.shape[0] == 0:
        matching_signals = signal.reshape((1,-1))
    else:
        matching_signals = np.vstack([matching_signals,signal])
    np.save('files/matching.npy', matching_signals)

###################################
# 5: PLOTTING THE SIGNALS
###################################

num_runs = int(poses.shape[0]/3)

# Summary plot for Matching Signals
subset_1 = matching_signals[types==0]
subset_2 = matching_signals[types==1]
subset_3 = matching_signals[types==2]

median_1, p25_1, p75_1 = compute_statistics(subset_1)
median_2, p25_2, p75_2 = compute_statistics(subset_2)
median_3, p25_3, p75_3 = compute_statistics(subset_3)

plt.figure(figsize=(12, 6))

plt.fill_between(
    range(subset_1.shape[1]),
    p25_1,
    p75_1,
    color='blue',
    alpha=0.3,
)
plt.plot(median_1, color='blue', label='HL')

plt.fill_between(
    range(subset_2.shape[1]),
    p25_2,
    p75_2,
    color='red',
    alpha=0.3,
)
plt.plot(median_2, color='red', label='ORCA')
plt.fill_between(
    range(subset_3.shape[1]),
    p25_3,
    p75_3,
    color='green',
    alpha=0.3,
)
plt.plot(median_3, color='green', label='SocialForce')
plt.title("Summary of Matching Signals")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.savefig("plots/matching_signals.png", dpi='figure', bbox_inches='tight')
