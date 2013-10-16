#!/usr/bin/env python
# CREATED:2013-10-14 10:18:29 by Brian McFee <brm2132@columbia.edu>
#  Do beat prediction from input spectrograms

import sys
import os
import glob
import argparse

import cPickle as pickle

import numpy as np
import scipy.stats
import librosa

from joblib import Parallel, delayed

SR      = 22050
N_FFT   = 2048
HOP     = 64
TIGHTNESS = 400

# Order: 
#   full, harmonic, percussive, lowrank, sparse
SPECMAP = {'full': 0, 'harmonic': 1, 'percussive': 2, 'lowrank': 3, 'sparse': 4}

def boxify(odf):

    odf = odf - odf.min()
    if odf.max() > 0:
        odf = odf / odf.max()
    return odf 

def onset_linear_sum(S, *args, **kwargs):

    return boxify(np.sum(np.maximum(0.0, np.diff(S, axis=1)), axis=0))

def onset_log_sum(S, *args, **kwargs):

    return onset_linear_sum(librosa.logamplitude(S / S.max()))

def onset_cbrt_sum(S, *args, **kwargs):

    return onset_linear_sum( (S / S.max()) ** (1./3) )
    
def onset_linear_quantile(S, q=0.5):

    D = np.maximum(0.0, np.diff(S, axis=1))
    return boxify(np.array(scipy.stats.mstats.mquantiles(D, [q], axis=0)).flatten())

def onset_log_quantile(S, *args, **kwargs):
    return onset_linear_quantile(librosa.logamplitude(S / S.max()), *args, **kwargs)

def onset_cbrt_quantile(S, *args, **kwargs):

    return onset_linear_quantile( (S / S.max())**(1./3), *args, **kwargs)

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

    parser.add_argument(    '-c',
                            '--cube-root',
                            dest        =   'cbrt',
                            required    =   False,
                            action      =   'store_true',
                            help        =   'Cube-root-scale the spectrogram')
    
    parser.add_argument(    '-l',
                            '--log',
                            dest        =   'log',
                            required    =   False,
                            action      =   'store_true',
                            help        =   'Log-scale the spectrogram')

    parser.add_argument(    '-q',
                            '--quantile',
                            dest        =   'quantile',
                            required    =   False,
                            type        =   float,
                            default     =   None,
                            help        =   'Quantile threshold for onsets')

    parser.add_argument(    '-s',
                            '--spectrogram',
                            dest        =   'spectrogram',
                            required    =   False,
                            choices     =   SPECMAP, 
                            default     =   'full',
                            help        =   'Spectrogram pre-processing')

    return vars(parser.parse_args(sys.argv[1:]))

def get_odf(**kw):

    odf = None

    if kw['log']:
        if kw['quantile'] is None:
            odf = onset_log_sum
        else:
            odf = onset_log_quantile
    elif kw['cbrt']:
        if kw['quantile'] is None:
            odf = onset_cbrt_sum
        else:
            odf = onset_cbrt_quantile
    else:
        if kw['quantile'] is None:
            odf = onset_linear_sum
        else:
            odf = onset_linear_quantile
    return odf

def process_file(input_file, **kwargs):

    output_file = os.path.basename(input_file)
    output_file = os.path.splitext(output_file)[0]

    if kwargs['log']:
        output_file = os.path.extsep.join([output_file, 'log'])
    elif kwargs['cbrt']:
        output_file = os.path.extsep.join([output_file, 'cbrt'])
    else:
        output_file = os.path.extsep.join([output_file, 'linear'])

    if kwargs['quantile'] is not None:
        output_file = os.path.extsep.join([output_file, 'q=%.2f' % kwargs['quantile']])
    else:
        output_file = os.path.extsep.join([output_file, 'sum'])
    
    output_file = os.path.extsep.join([output_file, kwargs['spectrogram']])
    output_file = os.path.extsep.join([output_file, 'csv'])
    output_file = os.path.join(kwargs['destination'], output_file)


    with open(input_file, 'r') as f:
        S = pickle.load(f)[SPECMAP[kwargs['spectrogram']]]
        Z = S.max()
        if Z > 0:
            S = S / Z

    onset = get_odf(**kwargs)
    tempo, beats = librosa.beat.beat_track( onsets=onset(S, q=kwargs['quantile']),
                                            sr=SR, n_fft=N_FFT, hop_length=HOP, tightness=TIGHTNESS)

    times = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    np.savetxt(output_file, times, fmt='%0.3f', delimiter='\n')
    pass

if __name__ == '__main__':
    
    parameters = process_args()
    files = sorted(glob.glob(parameters['input_glob']))


    Parallel(n_jobs=parameters['num_jobs'],
    verbose=5)(delayed(process_file)(input_file, **parameters) for input_file in files)
