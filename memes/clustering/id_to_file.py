import argparse
import itertools
from glob import glob
import gzip
from multiprocessing import Process, Manager, Pool
import re
from memes.utils import get_download_path, DATA_DIR, construct_output_filename

from tqdm import tqdm


def get_file_mapping(data):
    # get_downloaded(post_id, sub, month, url)
    ind, item = data
    post_id, sub, month, url = item
    filepath = get_download_path(post_id, sub, month, url)
    return post_id, str(filepath)


def main(args):
    manager = Manager()
    mapping = manager.dict()

    print(args.in_files)

    def iterator(in_file):
        try:
            month = re.search(r"\d{4}-\d{2}", in_file)[0]
        except:
            return
        with gzip.open(in_file, "rt") as f:
            for ind, line in enumerate(f):
                try:
                    post_id, title, sub, utc, url, user, _, _, _ = line.strip().split("\t")
                except:
                    pass
                yield ind, (post_id, sub, month, url)

    def data():
        return itertools.chain.from_iterable(
            [iterator(in_file) for in_file in args.in_files]
        )

    pool = Pool(args.num_procs)

    outfilename = construct_output_filename(subdir=DATA_DIR / "filepaths", prefix=args.prefix, suffix="id_to_file", ext="tsv")
    outfile = open(outfilename, "w")

    buf = []
    for ind, (post_id, filepath) in tqdm(
        enumerate(pool.imap(get_file_mapping, data(), chunksize=1_000_000))
    ):
        buf.append(post_id + "\t" + filepath + "\n")
        if ind % 100_000 == 0:
            outfile.write("".join(buf))
            buf = []
    outfile.write("".join(buf))
    buf = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_files", nargs="+", help="Path(s) to input .gz file")
    parser.add_argument(
        "--prefix", default=None, help="Prefix for output filename"
    )
    parser.add_argument(
        "--num_procs", default=4, type=int, help="Number of processes in pool"
    )
    main(parser.parse_args())
