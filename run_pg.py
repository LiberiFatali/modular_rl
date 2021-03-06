#!/usr/bin/env python
"""
This script runs a policy gradient algorithm
"""


from gym.envs import make
from modular_rl import *
import argparse, sys, cPickle
from tabulate import tabulate
import shutil, os, logging
import gym

import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    parser.add_argument("--env",required=True)
    parser.add_argument("--agent",required=True)
    parser.add_argument('--load_agent', required=False)
    parser.add_argument("--plot",action="store_true")
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])
    run = wandb.init()

    # contruct env
    if args.env == 'OsimGait':
        from osim.env import GaitEnv
        env = GaitEnv(visualize=False)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=env.spec.timestep_limit)
    elif args.env == 'OsimRun':
        from osim.env import RunEnv
        env = RunEnv(visualize=False)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=env.spec.timestep_limit)
    else:
        env = make(args.env)
    print 'env._max_episode_steps', env._max_episode_steps
    env_spec = env.spec
    mondir = args.outfile + ".dir"
    if os.path.exists(mondir): shutil.rmtree(mondir)
    os.mkdir(mondir)
    env = gym.wrappers.Monitor(env, mondir, video_callable=None if args.video else VIDEO_NEVER)

    # setup args and config
    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    args = parser.parse_args()
    # Force to env_spec.timestep_limit
    args.timestep_limit = env_spec.timestep_limit
    print 'args.timestep_limit', args.timestep_limit
    cfg = args.__dict__
    np.random.seed(args.seed)

    # construct agent
    if args.load_agent:
        agent = cPickle.load(open(args.load_agent))
    else:
        agent = agent_ctor(env.observation_space, env.action_space, cfg)

    if args.use_hdf:
        hdf, diagnostics = prepare_h5_file(args)
    gym.logger.setLevel(logging.WARN)

    run.config.update(args)

    COUNTER = 0
    def callback(stats):
        global COUNTER
        COUNTER += 1
        # Print stats
        print "*********** Iteration %i ****************" % COUNTER
        scalars = filter(lambda (k,v) : np.asarray(v).size==1, stats.items())
        print tabulate(scalars) #pylint: disable=W0110
        # Store to hdf5
        if args.use_hdf:
            for (stat,val) in stats.items():
                if np.asarray(val).ndim==0:
                    diagnostics[stat].append(val)
                else:
                    assert val.ndim == 1
                    diagnostics[stat].extend(val)
            if args.snapshot_every and ((COUNTER % args.snapshot_every==0) or (COUNTER==args.n_iter)):
                hdf['/agent_snapshots/%0.4i'%COUNTER] = np.array(cPickle.dumps(agent,-1))
            cPickle.dump(agent,
                    open(os.path.join(run.dir, 'agentsnapshot-%06i.pickle' % COUNTER), 'w'))
        run.history.add(dict(scalars))
        # Plot
        if args.plot:
            animate_rollout(env, agent, min(500, args.timestep_limit))

    run_policy_gradient_algorithm(env, agent, callback=callback, usercfg = cfg)

    if args.use_hdf:
        hdf['env_id'] = env_spec.id
        try: hdf['env'] = np.array(cPickle.dumps(env, -1))
        except Exception: print "failed to pickle env" #pylint: disable=W0703
    env.close()
