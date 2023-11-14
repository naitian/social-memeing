import argparse
from functools import partial
import itertools
from multiprocessing import Pool
from memes.utils import DATA_DIR, construct_output_filename

import imagehash
from PIL import Image

from tqdm import tqdm


def get_image_hash(data, hash_size):
    _, item = data
    post_id, filepath = item
    try:
        phash = imagehash.phash(Image.open(filepath), hash_size=hash_size)
        return post_id, str(phash)
    except:
        return post_id, None


def main(args):
    print(args.in_files)

    def iterator(in_file):
        with open(in_file, "rt") as f:
            for ind, line in enumerate(f):
                try:
                    post_id, filepath = line.strip().split("\t")
                except:
                    print(line)
                yield ind, (post_id, filepath)

    def data(skip=0):
        for i, d in enumerate(itertools.chain.from_iterable(
            [iterator(in_file) for in_file in args.in_files]
            )):
            if i >= skip:
                yield d


    pool = Pool(args.num_procs)

    outfilename = construct_output_filename(subdir=DATA_DIR / "imagehashes", prefix=args.prefix, suffix="hashes", ext="tsv")
    outfile = open(outfilename, "w")

    buf = []
    for ind, (post_id, phash) in tqdm(
        enumerate(pool.imap(partial(get_image_hash, hash_size=args.hash_size), data(args.skip), chunksize=500))
    ):
        if phash is None:
            continue
        buf.append(post_id + "\t" + phash + "\n")
        if ind % 100_000 == 0:
            outfile.write("".join(buf))
            buf = []
    outfile.write("".join(buf))
    buf = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_files", nargs="+", help="Path to input id-filepath map file(s)")
    parser.add_argument("--hash_size", default=8, type=int, help="Size of the hash")
    parser.add_argument(
        "--prefix", default=None, help="Prefix for the output filename"
    )
    parser.add_argument(
        "--num_procs", default=64, type=int, help="Number of processes in pool"
    )
    parser.add_argument(
        "--skip", default=0, type=int, help="Number of lines to skip"
    )
    main(parser.parse_args())
