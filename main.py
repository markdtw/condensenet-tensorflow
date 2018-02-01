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

    experiment = Experiment(args)
    experiment.train()
    print ('All Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages',
        type=str,
        default='14-14-14',
        help='stages.')
    parser.add_argument('--growth',
        type=str,
        default='8-16-32',
        help='growth rates.')
    # Child Net Hyperparameters
    parser.add_argument('--lr',
        type=float,
        default=1e-1,
        help='learning rate.')
    parser.add_argument('--momentum',
        type=float,
        default=9e-1,
        help='Momentum for SGD.')
    parser.add_argument('--ep',
        type=int,
        default=300,
        help='number of epochs.')
    parser.add_argument('--bsize',
        type=int,
        default=128,
        help='batch size.')
    # Logs
    parser.add_argument('--log-freq',
        type=int,
        default=100,
        help='Log every n iterations.')
    parser.add_argument('--model-dir',
        type=str,
        default='./log',
        help='Where to save the models.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0:
        raise SystemExit('Unknown argument: {}'.format(unparsed))
    main(args)
