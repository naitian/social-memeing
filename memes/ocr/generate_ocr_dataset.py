"""
Creates dataset with meme OCR output.

Each row contains:

post_id, subreddit, meme_template, meme_hash, ocr_output, img_path
"""

import argparse
import csv
import os
from pathlib import Path

from tqdm.auto import tqdm

from memes.utils import (DATA_DIR, IMAGE_DIR, OCR_DIR, HashClusters,
                         assert_dir, enclosing_dir, read_id_to_info, construct_output_filename)


def stream_ocr_dict(ocr_dir, img_dir):
    ocr_dir = Path(ocr_dir)
    for subreddit in tqdm(os.listdir(ocr_dir)):
        for month in os.listdir(ocr_dir / subreddit):
            with open(ocr_dir / subreddit / month, "r", encoding="utf-8") as f:
                for line in f:
                    fname, ocr_string = line.split("\t")
                    post_id = fname[:6]
                    try:
                        fpath = img_dir / subreddit / month / fname
                        result = (fpath, eval(ocr_string))
                        yield post_id, result
                    except ValueError:
                        print(fname)


def main(args):
    """
    Read in OCR outputs and write to file
    """
    assert_dir(enclosing_dir(args.outfile))
    print("reading cluster info")
    hashclusters = HashClusters(args.id_path, args.hash_path, args.cluster_path)
    filtered_clusters = {
        x for x in hashclusters.cluster_to_hash
        if len(hashclusters.filepaths_for_cluster(x)) > args.min_instances
    }

    print("reading subreddit info")
    post_info = read_id_to_info()
    print("streaming ocr output")

    outpath = construct_output_filename(
        subdir=DATA_DIR / "representations",
        prefix=args.prefix,
        suffix="filtered",
        ext="tsv",
    )
    with open(outpath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow([
            "post_id", "subreddit", "meme_template", "meme_hash", "ocr_output", "img_path"
            # "post_id", "meme_template", "meme_hash", "ocr_output", "img_path"
        ])
        for post_id, (fpath, ocr) in stream_ocr_dict(OCR_DIR, IMAGE_DIR):
            if post_id not in hashclusters.id_to_hash:
                # print(f"{post_id} not found in id_to_hash")
                continue
            phash = hashclusters.id_to_hash[post_id]
            if phash not in hashclusters.hash_to_cluster:
                print(f"{phash} not found in hash_to_cluster")
                continue
            template = hashclusters.hash_to_cluster[phash]
            if template not in filtered_clusters:
                continue

            subreddit = post_info[post_id].subreddit
            ocr = [ b[:2] for b in ocr ]  # strip confidence values if they exist

            writer.writerow([
                post_id,
                subreddit,
                template,
                phash,
                str(ocr),
                fpath
            ])
    print("done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("id_path", help="Path to the file mapping ID to filename")
    parser.add_argument("hash_path", help="Path to the file mapping hash to list of ID")
    parser.add_argument(
        "cluster_path", help="Path to the file mapping cluster ID to list of hashes"
    )
    parser.add_argument("--min_instances", type=int, default=50)
    parser.add_argument(
        "--prefix",
        type=str,
        help="Prefix",
        default="dataset",
    )
    main(parser.parse_args())
