#!/usr/bin/env python
# CREATED:2013-09-16 10:19:23 by Brian McFee <brm2132@columbia.edu>
#  Pre-compute spectrogram decompositions

import sys
import os
import glob
import argparse

import cPickle as pickle

import numpy as np
import librosa
import rpca

from joblib import Parallel, delayed



SR      = 22050
N_FFT   = 2048
HOP     = 64
N_MELS  = 128
FMAX    = 8000

WIN_P   = 19
WIN_H   = 19
HPSS_P  = 1.0

RPCA_MAX_ITER = 50

def process_audio(infile):

    y, sr = librosa.load(infile, sr=SR)

    # 1. Compute magnitude spectrogram
    D = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)).astype(np.float32)
    D = D / D.max()

    # 2. Compute mel filter bank
    M = librosa.feature.melfb(sr, N_FFT, n_mels=N_MELS, fmax=FMAX)
    M = M[:, :1+N_FFT/2].astype(np.float32)

    # 3. Compute HPSS

    Harm, Perc = librosa.hpss.hpss_median(D, win_H=WIN_H, win_P=WIN_P, p=HPSS_P)

    # 4. Compute RPCA
    Lowrank, Sparse, _ = rpca.robust_pca(D, max_iter=RPCA_MAX_ITER)

    Lowrank = np.maximum(0.0, Lowrank)
    Sparse  = np.maximum(0.0, Sparse)

    S       = M.dot(D)
    Harm    = M.dot(Harm)
    Perc    = M.dot(Perc)
    Lowrank = M.dot(Lowrank)
    Sparse  = M.dot(Sparse)

    return S, Harm, Perc, Lowrank, Sparse



def process_args():

    parser = argparse.ArgumentParser(description='Spectrogram decomposition')

    parser.add_argument( 'input_glob',
                         action = 'store',
                         help   = 'glob-string for files to process')

    parser.add_argument( '-j',
                         '--num_jobs',
                         dest   = 'num_jobs',
                         required   = False,
                         type       = int,
                         default    = 1,
                         help       = 'Number of parallel jobs')

    parser.add_argument( 'destination',
                         action = 'store',
                         help   = 'Path to store computed files')

    return vars(parser.parse_args(sys.argv[1:]))


def process_file(output_path, input_file):
    
    output_file = os.path.basename(input_file)
    output_file = os.path.splitext(output_file)[0] + os.path.extsep + 'pickle'
    output_file = os.path.join(output_path, output_file)

    data = process_audio(input_file)

    with open(output_file, 'w') as f:
        pickle.dump(data, f, protocol=-1)


if __name__ == '__main__':
    
    parameters = process_args()
    files = sorted(glob.glob(parameters['input_glob']))

    Parallel(n_jobs=parameters['num_jobs'], verbose=5)(delayed(process_file)(parameters['destination'], input_file) for input_file in files)

