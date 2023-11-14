"""
Before running this, generate a list of unique hashes using
```
cat 2020.tsv | cut -d$'\t' -f2 | sort | uniq > 2020-unique.tsv
```
"""
from numba import jit
import argparse
import itertools
from glob import glob
import gzip
from multiprocessing import Process, Manager, Pool
import re
from memes.utils import get_download_path, DATA_DIR, construct_output_filename
from memes.clustering.utils import to_int, to_binary_array

import torch
from PIL import Image

from tqdm import tqdm

import numpy as np


def hamming(a, b):
    dists = (
        torch.count_nonzero(~torch.eq(a.reshape(1, -1), b), axis=1).to("cpu").numpy()
    )
    # keep only the points where distance is less than a threshold
    keep = np.argwhere(dists <= 16)
    return dists[keep].reshape(-1), keep.reshape(-1)


def get_image_hash(data):
    ind, batch, hasharrays = data
    dists = hamming(hasharrays[ind], hasharrays[batch])
    keep = np.argwhere(dists <= 10)
    return ind, keep, dists[keep]


def main(args):
    manager = Manager()

    def iterator(in_file):
        with open(in_file, "rt") as f:
            for ind, line in enumerate(f):
                try:
                    phash = line.strip().split("\t")[1]
                    yield phash
                except:
                    print(line)

    
    def unique_hashes(hashes):
        seen = set()
        for phash in hashes:
            if phash not in seen:
                seen.add(phash)
                yield phash
    

    def binary_arrays(hashes):
        for phash in hashes:
            yield to_binary_array(to_int(phash), num_chars=len(phash))


    def data():
        outfilename = construct_output_filename(
            subdir=DATA_DIR / "imagehashes",
            prefix=args.prefix,
            suffix="unique",
            ext="tsv",
        )
        print("Creating unique hashes")
        unique = list(
                tqdm(unique_hashes(itertools.chain.from_iterable(
                    [iterator(in_file) for in_file in args.in_files]
                    )))
        )

        with open(outfilename, "w") as outfile:
            print("Writing unique hashes")
            for phash in tqdm(unique):
                outfile.write(phash + "\n")

        return np.array(list(binary_arrays(unique)))

    outfilename = construct_output_filename(
        subdir=DATA_DIR / "hammingdists",
        prefix=args.prefix,
        suffix="dists",
        ext="tsv",
    )
    outfile = open(outfilename, "w")
    last = 0

    buf = []
    hasharrays = torch.tensor(data()).to(device="cuda")
    for i in tqdm(range(len(hasharrays))):
        if i < last:
            continue
        dists, keep_ind = hamming(hasharrays[i], hasharrays[i:])
        for d, ki in zip(dists, keep_ind):
            buf.append(f"{i}\t{i+ki}\t{d}")
        if i % 1_000 == 0:
            outfile.write("\n".join(buf) + "\n")
            buf = []
    outfile.write("\n".join(buf) + "\n")
    buf = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_files", nargs="+", help="Path to input hash file(s)")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Prefix for output file name",
    )
    parser.add_argument(
        "--num_procs", default=64, type=int, help="Number of processes in pool"
    )
    main(parser.parse_args())
