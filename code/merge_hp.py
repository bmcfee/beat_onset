#!/usr/bin/env python
# CREATED:2013-10-17 13:00:54 by Brian McFee <brm2132@columbia.edu>
# stack harmonic and percussive components 


import sys
import glob
import argparse

import cPickle as pickle

import numpy as np

from joblib import Parallel, delayed


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

    return vars(parser.parse_args(sys.argv[1:]))


def process_file(input_file):
    
    with open(input_file, 'r') as f:
        SPECS = pickle.load(f)

    SPECS.append(np.vstack((SPECS[1], SPECS[2])))

    with open(input_file, 'w') as f:
        pickle.dump(SPECS, f, protocol=-1)


if __name__ == '__main__':
    
    parameters = process_args()
    files = sorted(glob.glob(parameters['input_glob']))

    Parallel(n_jobs=parameters['num_jobs'], verbose=5)(delayed(process_file)(input_file) for input_file in files)

