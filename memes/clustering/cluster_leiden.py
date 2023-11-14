"""
Run Leiden algorithm to generate clusters of hashes.
"""

import argparse
from array import array
import sys
from scipy.sparse import csr_matrix
from tqdm import tqdm
import math

import igraph as ig
import leidenalg as la

# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.neighbors._base import _check_precomputed, _is_sorted_by_data
import numpy as np
import pandas as pd

from memes.utils import read_year, DATA_DIR, construct_output_filename
from memes.clustering.utils import to_binary_array, to_int
from memes.clustering.clustering import read_distances, hash_to_ind, ind_to_hash


np.random.seed(0xB1AB)


def main(args):

    matrix = read_distances(args.distances, args.sample, threshold=args.threshold, dist_func=lambda x: x.max() + 1 - x, upper=True)
    hash_index = ind_to_hash(args.hash_index)

    print("clustering")
    graph = ig.Graph.Weighted_Adjacency(matrix, mode="upper")
    partitions = la.find_partition(
        graph,
        la.CPMVertexPartition,
        weights="weight",
        resolution_parameter=args.density,
        n_iterations=args.niters,
        seed=0xB1AB
    )

    outpath = construct_output_filename(
        subdir=DATA_DIR / "clusters",
        prefix=args.prefix,
        suffix="leiden",
        ext="tsv",
    )
    with open(outpath, "w") as f:
        for ind, cluster in enumerate(partitions):
            for hash_ind in cluster:
                phash = hash_index[hash_ind]
                f.write(f"{phash}\t{ind}\n")
    print(len(partitions), "clusters")
    quality = partitions.quality() / (2 * sum(graph.es["weight"]))
    print("Quality score of", quality)
    # print(clusters.value_counts())
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("distances")
    parser.add_argument("hash_index")
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--eps", type=float, default=8)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--min_samples", type=float, default=3)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--niters", type=int, default=2)
    parser.add_argument("--prefix", default=None)
    main(parser.parse_args(sys.argv[1:]))
