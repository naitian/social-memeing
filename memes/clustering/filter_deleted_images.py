"""
We filter out the 'This image was deleted'

See: 
- /shared/2/projects/meme-culture/data/images/dankmemes/RS_2020-08/i3ox28_9zsyznw051f51.jpg
"""
import argparse
from functools import partial
import itertools
from multiprocessing import Manager, Pool

import imagehash
from PIL import Image
from tqdm import tqdm

from memes.utils import DATA_DIR, construct_output_filename, list_images

deleted_images = open(DATA_DIR / "deleted_images.txt", "w")
extant_images = open(DATA_DIR / "extant_images.txt", "w")


def get_image_hash(data, references):
    try:
        i, line = data
        id, hash = line.strip().split("\t")
        phash = imagehash.hex_to_hash(hash)
        return phash in references, line
    except FileNotFoundError:
        return True, line


def main(args):
    pool = Pool(args.num_procs)
    reference_paths = [
        "/shared/2/projects/meme-culture/data/images/LeagueOfMemes/RS_2018-07/8vzgn9_QfsSQ9V.gifv",  # gifv version of the deleted image
        "/shared/2/projects/meme-culture/data/images/dankmemes/RS_2020-08/i3ox28_9zsyznw051f51.jpg",
        "/shared/2/projects/meme-culture/data/images/memes/RS_2019-05/bvapxx_xb1mtqi75l131.jpg",  # weird alternate version of the deleted image
    ]
    references = set([imagehash.phash( Image.open( reference_path), hash_size=args.hash_size) for reference_path in reference_paths])
    outfilename = construct_output_filename(
        subdir=DATA_DIR / "imagehashes", prefix=args.prefix, suffix="extant", ext="tsv"
    )
    outfile = open(outfilename, "w")

    def iterator(in_file):
        with open(in_file, "rt") as f:
            for ind, line in enumerate(f):
                yield ind, line

    def data(skip=0):
        for i, d in enumerate(
            itertools.chain.from_iterable(
                [iterator(in_file) for in_file in args.in_files]
            )
        ):
            if i >= skip:
                yield d

    buf = []
    for ind, (filter, line) in tqdm(
        enumerate(pool.imap(partial(get_image_hash, references=references), data(args.skip), chunksize=500))
    ):
        if filter:
            continue
        buf.append(line)
        if ind % 10_000 == 0:
            outfile.write("".join(buf))
            buf = []
    outfile.write("".join(buf))
    buf = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_files", help="Path to the input file(s) (e.g. imagehashes/2020.tsv)", nargs="+"
    )
    parser.add_argument("--prefix", default="2020", help="Prefix for output file")
    parser.add_argument("--hash_size", default=8, type=int, help="Hash size")
    parser.add_argument("--skip", default=0, type=int, help="Skip first N lines")
    parser.add_argument(
        "--num_procs", default=64, type=int, help="Number of processes in pool"
    )
    main(parser.parse_args())
