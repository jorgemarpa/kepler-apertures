import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import pickle

sys.path.append("%s/Work/BAERI/ADAP/kepler-apertures/" % os.environ["HOME"])
from kepler_apertures import EXBAMachine

parser = argparse.ArgumentParser(description="AutoEncoder")
parser.add_argument(
    "--quarter",
    dest="quarter",
    type=int,
    default=None,
    help="Which quarter.",
)
parser.add_argument(
    "--channel",
    dest="channel",
    type=int,
    default=1,
    help="List of files to be downloaded",
)
parser.add_argument(
    "--save",
    dest="save",
    action="store_true",
    default=True,
    help="Save models.",
)
args = parser.parse_args()


def run_code():
    q = args.quarter
    ch = args.channel
    root = "../data/export/%s" % (str(q))
    if not os.path.isdir(root):
        os.mkdir(root)

    obj_path = "../data/fits/exba/%i/%i/exba_object.pkl" % (q, ch)
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(
            "Quarter %i channel %i has no storaged files"
            % (q, ch)
        )
        nodata.append([q, ch])

    exba = pickle.load(open(obj_path, "rb"))

    exba.sources.to_csv("%s/exba_sources_channel_%02i.csv" % (root, ch))
    exba.period_df.to_csv("%s/bls_periods_channel_%02i.csv" % (root, ch))
    with open("%s/lcs_channel_%02i.pkl" % (root, ch), "wb") as f:
        pickle.dump({"lcs": exba.lcs,
                     "lcs_cbv": exba.corrected_lcs}, f)

if __name__ == '__main__':
    run_code()
