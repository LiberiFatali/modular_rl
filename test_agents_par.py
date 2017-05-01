#!/usr/bin/env python
"""
Load a snapshotted agent from an hdf5 file and animate it's behavior
"""

import argparse
import cPickle, h5py, numpy as np, time
from collections import defaultdict
import gym
import multiprocessing

def animate_rollout(env, agent, n_timesteps,delay=.01):
    infos = defaultdict(list)
    ob = env.reset()
    if hasattr(agent,"reset"): agent.reset()
    env.render()
    total_rew = 0
    for i in xrange(n_timesteps):
        ob = agent.obfilt(ob)
        a, _info = agent.act(ob)
        (ob, rew, done, info) = env.step(a)
        env.render()
        if done:
            print("terminated after %s timesteps"%i)
            break
        for (k,v) in info.items():
            infos[k].append(v)
        infos['ob'].append(ob)
        infos['reward'].append(rew)
        infos['action'].append(a)
        total_rew += rew
        print i, rew, total_rew
        time.sleep(delay)
    infos['total_reward'] = total_rew
    return infos

def worker(hdf_fname, snaps, out_queue):
    from osim.env import GaitEnv
    env = GaitEnv(visualize=False)

    hdf = h5py.File(hdf_fname, 'r')

    for s in snaps:
        snapname = '%04d' % s
        print 'SNAPNAME: %s' % snapname

        for i in xrange(10):
            np.random.seed(i)
            agent = cPickle.loads(hdf['agent_snapshots'][snapname].value)
            agent.stochastic=True
            infos = animate_rollout(env,agent, 500, delay=0)
            out_queue.put((snapname, infos['total_reward']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf")
    parser.add_argument("--env")
    parser.add_argument("--timestep_limit",type=int)
    args = parser.parse_args()

    # start processes

    proc_snaps = [
        range(900, 925),
        range(925, 950),
        range(950, 975),
        range(975, 1001)]
    queue = multiprocessing.Queue()
    for snaps in proc_snaps:
        proc = multiprocessing.Process(
            target=worker, args=(args.hdf, snaps, queue))
        proc.start()
    ofile = open('run_results-par.txt', 'a')
    while True:
        snapname, total_reward = queue.get()
        ofile.write('%s %s\n' % (snapname, total_reward))
        ofile.flush()

if __name__ == "__main__":
    main()
