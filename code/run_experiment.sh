#!/bin/bash

FEATURES=/q/porkpie/porkpie-p0/drspeech/data/beat/features
PREDICTIONS=~bmcfee/git/beat_onset/data/predictions_final
N_JOBS=16

for DATA in smc_dataset2 
do
    for spec in sparse full harmonic percussive lowrank 
    do
        for quant in '' '-m'
        do 
            ./beat_predictor.py $log $quant -s $spec -j $N_JOBS \
                $FEATURES/$DATA/\*.pickle \
                $PREDICTIONS/$DATA/
        done
    done
done
