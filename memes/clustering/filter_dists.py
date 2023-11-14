import argparse
import itertools
from collections import defaultdict
from glob import glob
import gzip
from multiprocessing import Process, Manager, Pool
import re
from utils import get_download_path, DATA_DIR
from pathlib import Path

from tqdm import tqdm


def main(args):
    if args.outfile is None:
        distfile = Path(args.hashes).stem
        outfile = open(DATA_DIR / "imagehashes" / f"{distfile}-filter_{args.threshold}.tsv", "w")
    else:
        outfile = open(args.outfile, "w")
    # First make a pass through the hashfile and count
    counts = defaultdict(int)
    for line in tqdm(open(args.hashes, "r")):
        _, phash = line.strip().split("\t")
        counts[phash] += 1
    # Now filter to threshold
    filtered = set([h for h in tqdm(counts) if counts[h] >= args.threshold])

    # Now filter the file (stream for memory's sake)
    buf = []
    for ind, line in tqdm(enumerate(open(args.hashes, "r"))):
        post_id, phash = line.strip().split("\t")
        if phash in filtered:
            buf.append(f"{post_id}\t{phash}\n")
        if ind % 100_000 == 0:
            outfile.write("".join(buf))
            buf = []
    outfile.write("".join(buf))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hashes", help="Path to the file mapping ID to hash")
    parser.add_argument("--threshold", default=10, type=int, help="Threshold to filter by (inclusive)")
    parser.add_argument(
        "--outfile", default=None, help="Path to the output file"
    )
    main(parser.parse_args())

