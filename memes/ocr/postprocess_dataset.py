"""
Post-processes the dataset to:
1) remove deleted posts
2) some OCR entries have a confidence score; remove these scores
3) generate file with just ocr output (to feed into tokenizer training)
"""

import argparse
import csv
import os
from pathlib import Path

from tqdm.auto import tqdm

from memes.utils import (DATA_DIR, IMAGE_DIR, OCR_DIR, HashClusters,
                         assert_dir, enclosing_dir, read_id_to_info)


def main(args):
    """
    Read in OCR outputs and write to file
    """
    assert_dir(enclosing_dir(args.outfile))
    assert_dir(enclosing_dir(args.rawtext))
    with open(args.dataset, "r", encoding="utf-8") as dataset, \
        open(args.rawtext, "w", encoding="utf-8") as rawtext, \
        open(args.outfile, "w", encoding="utf-8") as outfile:
        reader = csv.DictReader(dataset, delimiter="\t")
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        for row in tqdm(reader):
            if row["meme_template"] in args.skip_templates:
                continue
            rawtext.write(text + "\n")
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="Path to dataset.tsv",
        type=Path,
        default=DATA_DIR / "representations/dataset.tsv",
    )
    parser.add_argument(
        "--rawtext",
        help="Path to rawtext.txt",
        type=Path,
        default=DATA_DIR / "representations/rawtext.txt",
    )
    parser.add_argument(
        "--skip-templates",
        help="Templates to skip",
        default=["931_0"],
        nargs="+"
    )
    parser.add_argument(
        "--outfile",
        help="Path to processed.tsv",
        type=Path,
        default=DATA_DIR / "representations/processed.tsv",
    )
    main(parser.parse_args())
