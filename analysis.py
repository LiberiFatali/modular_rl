#!/bin/python

import collections
import sys

def read():
    data = open('run_results-par.txt')
    rows = []
    for line in data:
        snapname, reward = line.split()
        reward = float(reward)
        rows.append((snapname, reward))
    return rows

def threshold(data, thresh):
    return [r for r in data if r[1] > thresh]

def count(data):
    return collections.Counter([d[0] for d in data])

def main(argv):
    data = read()
    data = threshold(data, int(argv[1]))
    counted = count(data)
    for k in sorted(counted):
        print k, counted[k]

if __name__ == '__main__':
    main(sys.argv)
