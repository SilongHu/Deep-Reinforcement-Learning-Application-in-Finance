from __future__ import absolute_import
import json
import os
import time
from argparse import ArgumentParser
from datetime import datetime


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode",dest="mode",
                        help="train, generate, plot",
                        metavar="MODE", default="train")
    parser.add_argument("--choice", dest="choice",
                        help="choice the index of training subfolder",
                        default="1")

def main():
    parser = build_parser()
    options = parser.parse_args()
    # Package always be there.
    #if not os.path.exists("./" + "package"):
    #    os.makedirs("./" + "package")

    if not os.path.exists("./" + "database"):
        os.makedirs("./" + "database")

    if options.mode == "train":
        print ('training parts does not finish')
        #import pgportfolio.autotrain.training
        #if not options.algo:
        #    pgportfolio.autotrain.training.train_all(int(options.processes), options.device)
        #else:
        #    for folder in options.train_floder:
        #        raise NotImplementedError()
    elif options.mode == "generate":
        print ('Now focusing on generate data')
        print (options.choice)
        # This parts for downloading data into 'database'

    elif options.mode == "plot":
        print ('ploting parts does not finish')


if __name__ == "__main__":
    main()

