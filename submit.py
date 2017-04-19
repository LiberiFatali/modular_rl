import opensim as osim
from osim.http.client import Client
from osim.env import *
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
import numpy as np
import argparse
import h5py
import cPickle
import gym

def main():
    # Settings
    remote_base = 'http://grader.crowdai.org'

    # Command line parameters
    parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
    parser.add_argument("hdf")
    parser.add_argument('--token', dest='token', action='store', required=True)
    args = parser.parse_args()

    hdf = h5py.File(args.hdf,'r')

    env = GaitEnv(visualize=False)

    print hdf['agent_snapshots']['0947']
    agent = cPickle.loads(hdf['agent_snapshots']['0947'].value)
    agent.stochastic=False

    client = Client(remote_base)

    # Create environment
    observation = client.env_create(args.token)

    total_reward = 0
    # Run a single step
    for i in range(501):
        ob = agent.obfilt(observation)
        a, _info = agent.act(ob)
        [observation, reward, done, info] = client.env_step(a.tolist(), True)
        print i, reward, done
        total_reward += reward
        if done:
            break

    print 'TOTAL REWARD: ', total_reward
    raw_input('press ENTER to submit')
    client.submit()

if __name__ == '__main__':
    main()
