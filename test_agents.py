#!/usr/bin/env python
"""
Load a snapshotted agent from an hdf5 file and animate it's behavior
"""

import argparse
import cPickle, h5py, numpy as np, time
from collections import defaultdict
import gym

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf")
    parser.add_argument("--env")
    parser.add_argument("--timestep_limit",type=int)
    args = parser.parse_args()

    hdf = h5py.File(args.hdf,'r')

    snapnames = hdf['agent_snapshots'].keys()
    print "snapshots:\n",snapnames

    from osim.env import GaitEnv
    env = GaitEnv(visualize=False)

    ofile = open('run_results.txt', 'a')
    for i in xrange(900, 1000):
        snapname = '%04d' % i
        print 'SNAPNAME: %s' % snapname

        for i in xrange(10):
            agent = cPickle.loads(hdf['agent_snapshots'][snapname].value)
            agent.stochastic=False
            infos = animate_rollout(env,agent, 500, delay=0)
            ofile.write('%s %s\n' % (snapname, infos['total_reward']))
            ofile.flush()


if __name__ == "__main__":
    main()
