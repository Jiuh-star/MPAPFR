#!/usr/bin/env bash

source .venv/bin/activate

REPEATS=5
TEMPLATE=configs/template.in.toml
SETTING=cifar10-cnn  # cifar10-cnn, cifar10-resnet18, femnist-cnn, cifar100-cnn
WORKDIR="workdir"

# code the following with AI

# Prompts:
# FedAvgFT, FedProxFT, pFedMe, FedBN, FedRep, FedALA, FedSelect, FedAS, Ditto, GPFL
# FedAvg, FedProx

set -ex

# debug
# seed=1
# pfl run -c $TEMPLATE -o seed=$seed -o protocol=FedAvgFT -o workdir=$WORKDIR/$SETTING/FedAvgFT/$seed/

# fix memory leak
# export OMP_NUM_THREADS=4
# export MKL_NUM_THREADS=4

# main
for seed in `seq 1 $REPEATS`;
do
    pfl run -c $TEMPLATE -o seed=$seed -o protocol=FedAvgFT -o workdir=$WORKDIR/$SETTING/FedAvgFT/$seed/ &
    sleep 1  # for weird skipped error
    # wait

    pfl run -c $TEMPLATE -o seed=$seed -o protocol=Ditto -o workdir=$WORKDIR/$SETTING/Ditto/$seed/ \
                                       -o client.personalized_optimizer.type=PerturbedGradientDescent \
                                       -o client.personalized_optimizer.mu=1 \
                                       &> /dev/null &
    sleep 1
    wait
done

set +ex