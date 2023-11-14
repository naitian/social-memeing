"""
Generate clusters of memes based on their semantic embeddings.
"""

import argparse
from collections import defaultdict

import igraph as ig
import leidenalg as la
import numpy as np
import torch
from datasets import load_from_disk
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from transformers import AutoConfig, AutoModelForSequenceClassification

from memes.utils import DATA_DIR, HashClusters, construct_output_filename


def main(args):
    raw_dataset = load_from_disk(DATA_DIR / "representations/all-8-processed-dataset")
    dataset = raw_dataset.remove_columns(['post_id', 'subreddit', 'meme_hash', 'ocr_output', 'img_path'])
    dataset = dataset.rename_column("meme_template", "labels")
    num_labels = dataset["train"].features["labels"].num_classes

    config = AutoConfig.from_pretrained("roberta-base", num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_config(config)

    state_dict = torch.load(DATA_DIR / "representations/classifier/checkpoints/c-roberta_pt_lowlr-pretrain-20230719-191731/best.pt")
    model.load_state_dict(state_dict)
    model.to("cpu")

    # hc = HashClusters(
    #     DATA_DIR / "filepaths/all.tsv",
    #     DATA_DIR / "imagehashes/all-8-processed-hashes.tsv",
    #     DATA_DIR / "clusters/all-8-processed-leiden.tsv"
    # )


    embeddings = model.classifier.out_proj.weight.cpu().detach().numpy()
    embeddings = (embeddings - embeddings.mean(axis=0, keepdims=True)) / embeddings.std(axis=0, keepdims=True)

    cos_sim = embeddings @ embeddings.T / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embeddings, axis=1).reshape(-1, 1))

    # pca = PCA(n_components=64)
    # x = pca.fit_transform(embeddings)

    # cos_sim = x @ x.T / (np.linalg.norm(x, axis=1) * np.linalg.norm(x, axis=1).reshape(-1, 1))

    num = args.num
    _, inds = torch.topk(torch.tensor(cos_sim), num + 1, dim=0)
    cos_sim = np.zeros_like(cos_sim)
    cos_sim[inds[0], inds] = (0.9 ** np.arange(num + 1)).reshape(-1, 1)

    print(cos_sim.mean())
    print(np.sum(cos_sim > 0))

    graph = ig.Graph.Weighted_Adjacency(csr_matrix(cos_sim), mode="max")
    partitions = la.find_partition(graph, la.CPMVertexPartition, weights="weight", resolution_parameter=0.1)

    outpath = construct_output_filename(
        subdir=DATA_DIR / "semantic_clusters",
        prefix=args.prefix,
        suffix=f"clusters-norm-{args.num}",
        ext="tsv",
    )
    with open(outpath, "w", encoding="utf-8") as f:
        for ind, cluster in enumerate(partitions):
            for tpl_ind in cluster:
                tpl = dataset["train"].features["labels"].int2str(tpl_ind)
                f.write(f"{tpl}\t{ind}\n")
    print(len(partitions), "clusters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="all-8-processed")
    parser.add_argument("--num", default=50)
    main(parser.parse_args())
