#!/bin/sh

env=$1
load=$2

export KERAS_BACKEND=theano

xvfb-run -s "-screen 0 1400x900x24" \
    ./run_pg.py \
        --env=$env \
        --outfile=/root/share/$env.h5 \
        --agent modular_rl.agentzoo.TrpoAgent \
        --load_agent $2 \
        --gamma=0.995 \
        --lam=0.97 \
        --max_kl=0.01 \
        --cg_damping=0.1 \
        --activation=tanh \
        --seed=0 \
        --use_hdf=1 \
        --snapshot_every=1 \
        --timesteps_per_batch=50000 \
        --parallel=0 \
        --profile=0 \
        --video=0 \
        --n_iter=1000
