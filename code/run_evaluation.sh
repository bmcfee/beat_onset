#!/bin/bash

TRUTH=/q/porkpie/porkpie-p0/drspeech/data/beat/features
PREDICTIONS=~bmcfee/git/beat_onset/data/predictions
OUTPUT=~bmcfee/git/beat_onset/data/results
N_JOBS=16

for DATA in rwcj jazz_roger mirex2006 smc_dataset2 
do
    ./beat_evaluator.py -j $N_JOBS \
        $PREDICTIONS/$DATA/\*.csv \
        $TRUTH/$DATA/ \
        $OUTPUT/$DATA/
done
