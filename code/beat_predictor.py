#!/usr/bin/env python
# CREATED:2013-10-14 10:18:29 by Brian McFee <brm2132@columbia.edu>
#  Do beat prediction from input spectrograms

import sys
import os
import glob
import argparse

import cPickle as pickle

import numpy as np
import librosa

from joblib import Parallel, delayed

SR      = 22050
N_FFT   = 2048
HOP     = 64
TIGHTNESS = 400

MED_SIZE = 5

# Order: 
#   full, harmonic, percussive, lowrank, sparse
SPECMAP = {'full': 0, 'harmonic': 1, 'percussive': 2, 'lowrank': 3, 'sparse': 4, 'hp': 5}

def process_args():
    
    parser = argparse.ArgumentParser(description='Beat tracking predictor')

    parser.add_argument(    'input_glob',
                            action      =   'store',
                            help        =   'glob-string for files to process')

    parser.add_argument(    '-j',
                            '--num_jobs',
                            dest        =   'num_jobs',
                            required    =   False,
                            type        =   int,
                            default     =   1,
                            help        =   'Number of parallel jobs')

    parser.add_argument(    'destination',
                            action      =   'store',
                            help        =   'Path to store computed files')

    parser.add_argument(    '-m',
                            '--median',
                            dest        =   'median',
                            required    =   False,
                            action      =   'store_true',
                            help        =   'median-filter the spectrogram')

    parser.add_argument(    '-s',
                            '--spectrogram',
                            dest        =   'spectrogram',
                            required    =   False,
                            choices     =   SPECMAP, 
                            default     =   'full',
                            help        =   'Spectrogram pre-processing')

    return vars(parser.parse_args(sys.argv[1:]))

def process_file(input_file, **kwargs):

    output_file = os.path.basename(input_file)
    output_file = os.path.splitext(output_file)[0]
    output_file = os.path.extsep.join([output_file, 'log'])

    if kwargs['median']:
        output_file = os.path.extsep.join([output_file, 'med'])
    else:
        output_file = os.path.extsep.join([output_file, 'sum'])
    
    output_file = os.path.extsep.join([output_file, kwargs['spectrogram']])
    output_file = os.path.extsep.join([output_file, 'csv'])
    output_file = os.path.join(kwargs['destination'], output_file)

    with open(input_file, 'r') as f:
        S = pickle.load(f)[SPECMAP[kwargs['spectrogram']]].astype(np.float32)

    if kwargs['median']:
        odf = librosa.onset.onset_strength(S=S, sr=SR, hop_length=HOP, n_fft=N_FFT, aggregate=np.median)
    else:
        odf = librosa.onset.onset_strength(S=S, sr=SR, hop_length=HOP, n_fft=N_FFT, aggregate=np.mean)

    tempo, beats = librosa.beat.beat_track( onsets=odf, sr=SR, hop_length=HOP, tightness=TIGHTNESS)

    times = librosa.frames_to_time(beats, sr=SR, hop_length=HOP, n_fft=N_FFT)
    librosa.output.times_csv(output_file, times)

if __name__ == '__main__':
    
    parameters = process_args()
    files = sorted(glob.glob(parameters['input_glob']))


    Parallel(n_jobs=parameters['num_jobs'],
    verbose=5)(delayed(process_file)(input_file, **parameters) for input_file in files)
