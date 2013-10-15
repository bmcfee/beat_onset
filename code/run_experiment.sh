#!/bin/bash

FEATURES=/q/porkpie/porkpie-p0/drspeech/data/beat/features
PREDICTIONS=~bmcfee/git/beat_onset/data/predictions
N_JOBS=16

for DATA in rwcj jazz_roger mirex2006 smc_dataset2 
do
    for spec in sparse full harmonic percussive lowrank 
    do
        for log in '' -l
        do
            for quant in '' '-q 0.25' '-q 0.5' '-q 0.75'
            do 
                ./beat_predictor.py $log $quant -s $spec -j $N_JOBS \
                    $FEATURES/$DATA/\*.pickle \
                    $PREDICTIONS/$DATA/
            done
        done
    done
done
