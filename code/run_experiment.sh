#!/bin/bash

FEATURES=/q/porkpie/porkpie-p0/drspeech/data/beat/features
PREDICTIONS=/home/bmcfee/git/beat_onset/data/predictions
N_JOBS=16

for DATA in mirex2006 rwcj jazz_roger smc_dataset2 
do
    for spec in full harmonic percussive lowrank sparse
    do
        for log in -l ''
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
