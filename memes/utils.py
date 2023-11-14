from collections import defaultdict, namedtuple
import os
from glob import glob
import gzip
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd

from tqdm import tqdm


DATA_DIR = Path("/shared/2/projects/meme-culture/data/")
IMAGE_DIR = Path("/shared/2/projects/meme-culture/data/images/")
OCR_DIR = Path("/shared/2/projects/meme-culture/data/ocr/")
HASH_DIR = Path("/shared/2/projects/meme-culture/data/hashed-data/")
MODEL_DIR = Path("/shared/2/projects/meme-culture/model/")


Post = namedtuple(
    "Post",
    ("post_id", "title", "subreddit", "utc", "url", "user", "score", "num_comments", "is_self")
)


### Script utils


def assert_dir(dir):
    os.makedirs(dir, exist_ok=True)


def filename(path):
    """Return just the name of the file.

    e.g. /path/to/file.txt -> file
    """
    return os.path.splitext(os.path.basename(path))[0]


def enclosing_dir(path):
    """Return the directory containing the file.

    e.g. /path/to/file.txt -> /path/to
    """
    return os.path.dirname(path)


def construct_output_filename(subdir=None, prefix=None, suffix=None, ext=None, include_timestamp=False):
    """Constructs a filename for an output file with a timestamp.

    Args:
        subdir: The subdirectory to put the file in.
        prefix: The prefix for the filename.
        suffix: The suffix for the filename.
        ext: The extension for the filename.

    Returns:
        A string representing the output filename.
    """
    if subdir is not None:
        assert_dir(subdir)
    if include_timestamp:
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = timestamp + "-"
    else:
        filename = ""
    if prefix is not None:
        filename = prefix + "-" + filename
    if suffix is not None:
        filename = filename + suffix
    if ext is not None:
        filename = filename + "." + ext
    if subdir is not None:
        filename = subdir / filename
    return filename


###


def read_id_to_info():
    files = glob(str(DATA_DIR / "*.gz"))
    df = {}
    for f in tqdm(files):
        print(f)
        with gzip.open(f, "rt") as f:
            for line in f:
                try:
                    post_id, title, sub, utc, url, user, score, num_comments, is_self = line.strip().split("\t")
                    df[post_id] = Post(post_id, title, sub, int(utc), url, user, int(score), int(num_comments), is_self)
                except:
                    pass
    return df


def read_id_to_path(path):
    """Maps from post ID to file path."""
    id_to_path = dict()
    with open(path, "r") as f:
        for line in tqdm(f):
            post_id, path = line.strip().split("\t")
            id_to_path[post_id] = path
    return id_to_path


def read_hash_to_ids(path, reverse=False):
    """Maps from hash to a list of post IDs"""
    hash_to_ids = defaultdict(list)
    id_to_hash = dict()
    with open(path, "r") as f:
        for line in tqdm(f):
            post_id, phash = line.strip().split("\t")
            hash_to_ids[phash].append(post_id)
            if reverse:
                id_to_hash[post_id] = phash
    if reverse:
        return hash_to_ids, id_to_hash
    return hash_to_ids


def read_hash_clusters(path):
    """Maps from cluster ID to a list of hashes"""
    hash_to_cluster = dict()
    cluster_to_hash = defaultdict(list)
    with open(path, "r") as f:
        for line in tqdm(f):
            phash, cluster = line.strip().split("\t")
            hash_to_cluster[phash] = cluster
            cluster_to_hash[cluster].append(phash)
    return hash_to_cluster, cluster_to_hash


class HashClusters:
    def __init__(self, id_path, hash_path, cluster_path):
        self.id_to_path = read_id_to_path(id_path)
        self.hash_to_ids, self.id_to_hash = read_hash_to_ids(hash_path, reverse=True)
        self.hash_to_cluster, self.cluster_to_hash = read_hash_clusters(cluster_path)

    def filepaths_for_cluster(self, cluster):
        """Returns a list of filepaths for a given cluster."""
        hashes = self.cluster_to_hash[cluster]
        filepaths = [
            self.id_to_path[post_id]
            for phash in hashes
            for post_id in self.hash_to_ids[phash]
            if is_img(self.id_to_path[post_id])
        ]
        return filepaths

    def filepaths_for_hash(self, phash):
        """Returns a list of filepaths for a given hash."""
        filepaths = [
            self.id_to_path[post_id]
            for post_id in self.hash_to_ids[phash]
            if is_img(self.id_to_path[post_id])
        ]
        return filepaths


###


def is_img(url):
    return url[-3:] in set(["png", "jpg"])


def list_images(subset=False, cache=True):
    cache_file = ["images.txt", "images_subset.txt"][subset]
    if not cache:
        raise NotImplementedError
    for line in open(DATA_DIR / cache_file):
        yield line.strip()


def read_data(month):
    filepath = f"/shared/2/projects/meme-culture/data/hashed-data/RS_{month}-hashed.tsv"
    df = pd.read_csv(filepath, delimiter="\t")
    df["month"] = month
    return df


def read_year(year):
    dfs = []

    for i in range(1, 13):
        month = f"{year}-{i:02}"
        print(f"Reading {month}")
        dfs.append(read_data(month))
    return pd.concat(dfs)


def get_download_path(post_id, sub, month, url):
    filename = os.path.basename(urlparse(url).path)
    img_path = IMAGE_DIR / sub / f"RS_{month}" / f"{post_id}_{filename}"
    return img_path
