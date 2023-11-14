"""
Run DBSCAN to generate clusters of hashes.
"""

import argparse
from array import array
import sys
from scipy.sparse import csr_matrix
from tqdm import tqdm
import math

# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.neighbors._base import _check_precomputed, _is_sorted_by_data
import numpy as np
import pandas as pd

from memes.utils import read_year, DATA_DIR, construct_output_filename
from memes.clustering.utils import to_binary_array, to_int


def read_distances(path, sample=None, return_csr=True, threshold=10, dist_func=lambda x: x, keeplist=None, upper=False):
    data = array("I")
    rows = array("I")
    cols = array("I")
    # try only taking pairs w distance <= 10
    # per zsavvas
    THRESHOLD = threshold

    class np_buffer_wrapper:
        """
        Create an array interface for numpy so we can directly refer to
        memory location.
        """
        def __init__(self, ptr, shape, typestr):
            self.__array_interface__ = {
                "shape": shape,
                "typestr": typestr,
                "data": (ptr, True),
            }

        @classmethod
        def from_array(cls, array):
            endianness = {"little": "<", "big": ">"}
            ptr, size = array.buffer_info()
            byteorder = endianness[sys.byteorder]
            # TODO: right now we assume unsigned int. best to infer from the
            # array
            basictype = "u"
            numbytes = array.itemsize
            typestr = byteorder + basictype + str(numbytes)
            return cls(ptr, (size,), typestr)

    def add_row(ind1, ind2, dist):
        nonlocal data
        nonlocal rows
        nonlocal cols
        nonlocal keeplist
        if keeplist is not None:
            if not (int(ind1) in keeplist and int(ind2) in keeplist):
                return
            ind1 = keeplist[int(ind1)]
            ind2 = keeplist[int(ind2)]
        if int(ind1) > sample or int(ind2) > sample:
            return
        if int(dist) > THRESHOLD:
            return
        if return_csr:
            if upper:
                data.extend([int(dist)])
                rows.extend([int(ind1)])
                cols.extend([int(ind2)])
            else:
                data.extend([int(dist), int(dist)])
                rows.extend([int(ind1), int(ind2)])
                cols.extend([int(ind2), int(ind1)])

    print("reading distances")
    if sample is None:
        sample = math.inf
    with open(path, "r") as f:
        for i, line in tqdm(enumerate(f)):
            if i > sample:
                break
            try:
                ind1, ind2, dist = line.strip().split("\t")
                add_row(ind1, ind2, dist)
            except Exception as e:
                # NOTE: this exception handling addresses a bug in distance
                # calculations that has since been fixed.
                pass
            #     ind1, ind2, dist_ind3, ind4, dist2 = line.strip().split("\t")
            #     dist = dist_ind3[: -len(ind1)]
            #     add_row(ind1, ind2, dist)

            #     ind3 = dist_ind3[len(dist) :]
            #     add_row(ind3, ind4, dist2)
        d = np.array(np_buffer_wrapper.from_array(data), copy=False)
        r = np.array(np_buffer_wrapper.from_array(rows), copy=False)
        c = np.array(np_buffer_wrapper.from_array(cols), copy=False)
        d = dist_func(d)
        if return_csr:
            print("constructing matrix")
            return csr_matrix((d, (r, c)))
        return np.stack([d, r, c])


def hash_to_ind(path):
    """path to file of unique hashes.

    same as path being passed into the distance calculation.
    """
    hashes = {}
    print("reading index")
    with open(path, "r") as f:
        for i, line in tqdm(enumerate(f)):
            hashes[line.strip()] = i
    return hashes


def ind_to_hash(path):
    """path to file of unique hashes.

    same as path being passed into the distance calculation.
    """
    hashes = list()
    print("reading index")
    with open(path, "r") as f:
        for i, line in tqdm(enumerate(f)):
            hashes.append(line.strip())
    return hashes


np.random.seed(0xB1AB)


def main(args):

    distances = read_distances(args.distances, args.sample)
    # distances = _check_precomputed(distances)
    hash_index = ind_to_hash(args.hash_index)

    print("clustering")

    dbscan = DBSCAN(metric="precomputed", eps=args.eps, min_samples=args.min_samples, n_jobs=16)
    # try using OPTICS instead for lower memory
    # dbscan = OPTICS(metric="precomputed", min_samples=args.min_samples)

    dbscan.fit(distances)
    clusters = pd.Series(dbscan.labels_, index=list(range(distances.shape[0])))
    cluster_dict = clusters.to_dict()
    print("writing output file")

    outpath = construct_output_filename(
        subdir=DATA_DIR / "clusters",
        prefix=args.prefix,
        suffix="clusters",
        ext="tsv",
    )
    with open(outpath, "w") as f:
        for ind, cluster in cluster_dict.items():
            phash = hash_index[ind]
            f.write(f"{phash}\t{cluster}\n")
    print(clusters.value_counts())
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("distances")
    parser.add_argument("hash_index")
    parser.add_argument("--eps", type=float, default=8)
    parser.add_argument("--min_samples", type=float, default=3)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--prefix", default=None)
    main(parser.parse_args(sys.argv[1:]))
