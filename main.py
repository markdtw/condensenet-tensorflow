# py2 - py3 compatibility settings
from __future__ import absolute_import, division, print_function, unicode_literals
# build-in libraries
import os
import pdb
import argparse
# installed libraries
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# local files
from experiment import Experiment

def main(args):
    """Main function"""

    experiment = Experiment(args, (not args.evaluate))
    experiment.run()
    print ('All Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate',
        type=bool,
        default=False,
        help='Evaluate model.')
    # Hyperparameters
    parser.add_argument('--lr',
        type=float,
        default=1e-1,
        help='learning rate.')
    parser.add_argument('--ep',
        type=int,
        default=300,
        help='number of epochs.')
    parser.add_argument('--bsize',
        type=int,
        default=128,
        help='batch size.')
    # Log configs
    parser.add_argument('--model-path',
        type=str,
        default=None,
        help='Pre-trained model path.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0:
        raise SystemExit('Unknown argument: {}'.format(unparsed))
    main(args)
