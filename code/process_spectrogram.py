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

# HPSS Parameters
KERNEL_SIZE = 31
HPSS_P      = 2.0

RPCA_MAX_ITER = 50

def hpss(y):

    D = librosa.stft(y)
    H, P = librosa.decompose.hpss(D, kernel_size=KERNEL_SIZE, power=HPSS_P)

    D_harm = np.abs(librosa.stft(librosa.istft(H), n_fft=N_FFT, hop_length=HOP))
    D_perc = np.abs(librosa.stft(librosa.istft(P), n_fft=N_FFT, hop_length=HOP))

    return D_harm, D_perc

def process_audio(infile):

    y, sr = librosa.load(infile, sr=SR)

    # 1. Compute magnitude spectrogram
    D = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))

    # 2. Compute HPSS
    Harm, Perc = hpss(y)

    # 3. Compute RPCA
    Lowrank, Sparse, _ = rpca.robust_pca(D, max_iter=RPCA_MAX_ITER)

    Lowrank = np.maximum(0.0, Lowrank)
    Sparse  = np.maximum(0.0, Sparse)

    S       = librosa.feature.melspectrogram(librosa.logamplitude(D, ref_power=D.max()), 
                                             sr=sr,
                                             n_mels=N_MELS,
                                             fmax=FMAX)

    Harm       = librosa.feature.melspectrogram(librosa.logamplitude(Harm, ref_power=Harm.max()), 
                                             sr=sr,
                                             n_mels=N_MELS,
                                             fmax=FMAX)

    Perc       = librosa.feature.melspectrogram(librosa.logamplitude(Perc, ref_power=Perc.max()), 
                                             sr=sr,
                                             n_mels=N_MELS,
                                             fmax=FMAX)

    Lowrank       = librosa.feature.melspectrogram(librosa.logamplitude(Lowrank, ref_power=Lowrank.max()), 
                                             sr=sr,
                                             n_mels=N_MELS,
                                             fmax=FMAX)

    Sparse       = librosa.feature.melspectrogram(librosa.logamplitude(Sparse, ref_power=Sparse.max()), 
                                             sr=sr,
                                             n_mels=N_MELS,
                                             fmax=FMAX)

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

