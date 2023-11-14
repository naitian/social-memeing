"""
Remove texts that are common across a given template. Also remove templates that
have fewer than min_instances number of instances.

We do this, parallelizing across templates.

First, we read in the OCR data.
"""

import argparse
from collections import defaultdict
import csv
from functools import partial
from pathlib import Path
from multiprocessing import Pool

from tqdm.auto import tqdm

from memes.utils import DATA_DIR, construct_output_filename


def read_dataset(path):
    """
    Read OCR dataset into a template -> row map.
    """
    templates = defaultdict(list)
    with open(path, "r", encoding="utf-8") as dataset:
        reader = csv.DictReader(dataset, delimiter="\t")
        for row in tqdm(reader):
            templates[row["meme_template"]].append(row)
    return templates


def filter_template(template_id, instances, threshold=0.9, min_instances=50):
    """
    We white-space tokenize and filter out any words from the OCR output that
    exist in greater than <threshold> proportion of the meme instances.

    The min_ct parameter specifies the number of memes a fill word must appear
    in for it to be filtered, regardless of the threshold.

    Because we filter at the word level, the bounding boxes will not correspond
    exactly to the words. We accept this compromise because we only use BBs
    coarsely for ordering.

    Iterate over all instances; maintain a running count of box fills.

    Calculate which box fills occur in over THRESHOLD number of memes.

    Iterate over all instances again and filter out those box fills.
    """
    if len(instances) < min_instances:
        return (template_id, [])

    # First loop: count fills
    fill_cts = defaultdict(int)
    for instance in instances:
        ocr = eval(instance["ocr_output"])
        instance_words = set()
        for box in ocr:
            instance_words.update(box[1].lower().strip().split())
        for word in instance_words:
            fill_cts[word] += 1

    min_counts = int(threshold * len(instances))
    to_filter = set([
        text for text in fill_cts if fill_cts[text] > min_counts
    ])

    def _process_box(box):
        """
        Strip filtered words from the box
        """
        return [
            box[0],
            " ".join([word for word in box[1].strip().split() if word.lower() not in to_filter])
        ]

    # Second loop: filter fills
    new_rows = []
    for instance in instances:
        # remove instances that have no fills
        if len(instance) == 0:
            continue
        ocr = eval(instance["ocr_output"])

        ocr = [
            result for box in ocr if len((result := _process_box(box))[1]) > 0
        ]
        instance["ocr_output"] = ocr
        if len(ocr) > 0:
            assert len(instance["ocr_output"]) > 0, "OCR output cannot be empty"
            new_rows.append(instance)

    return (template_id, new_rows)


def write_output(results, prefix):
    """
    Write output to new file.
    """
    outpath = construct_output_filename(
        subdir=DATA_DIR / "representations",
        prefix=prefix,
        suffix="filtered",
        ext="tsv",
    )
    fieldnames = results[0][1][0].keys()
    with open(outpath, "w", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for _, rows in results:
            for row in rows:
                writer.writerow(row)


def main(args):
    """
    First, we read the OCR data.

    Then, parallelizing across templates, we filter out boxes for each template.

    Finally, we aggregate all the results and write out a new file with the same
    fields, but with common OCR boxes removed.
    """
    templates = read_dataset(args.dataset)
    with Pool(processes = args.num_processes) as pool:
        map_fn = partial(filter_template, threshold=args.threshold, min_instances=args.min_instances)
        results = [r for r in pool.starmap(map_fn, templates.items()) if len(r[1]) > 0]
        write_output(results, args.prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATA_DIR / "representations/dataset.tsv")
    parser.add_argument("--min_instances", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--prefix", type=str, default="dataset")
    parser.add_argument("--num_processes", type=int, default=4)

    main(parser.parse_args())
