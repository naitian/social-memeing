"""
Generate clusters of memes based on their semantic embeddings.
"""

import argparse
from collections import defaultdict
import random
from tqdm.auto import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor
from transformers import CLIPModel, CLIPProcessor
from datasets import load_from_disk
import igraph as ig
import leidenalg as la
import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from PIL import Image


from memes.utils import DATA_DIR, HashClusters, construct_output_filename

device = "cuda"

def main(args):
    random.seed(0xb1ab)
    # raw_dataset = load_from_disk(DATA_DIR / "representations/all-8-processed-clip")
    # dataset = raw_dataset.remove_columns(['post_id', 'subreddit', 'meme_hash', 'ocr_output', 'img_path'])
    # dataset = dataset.rename_column("meme_template", "labels")
    # num_labels = dataset["train"].features["labels"].num_classes

    model = CLIPModel.from_pretrained("../data/representations/clip/checkpoints/clip-pretrain-20230825-035745-bf16/best/")
    model.to(device)

    basemodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    basemodel.to(device)


    dataset = load_from_disk(DATA_DIR / "representations/all-8-processed-clip")
    dataset = dataset.remove_columns(['post_id', 'subreddit', 'meme_hash', 'ocr_output', 'img_path','resampled_img_path'])
    tpls = dataset["train"].unique("meme_template")
    # num_labels = dataset["train"].features["labels"].num_classes


    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    hc = HashClusters(
        "../data/filepaths/all.tsv",
        "../data/imagehashes/all-8-processed-hashes.tsv",
        "../data/clusters/all-8-processed-leiden.tsv"
    )

    def embed_image(img_paths):
        embedding = None
        with torch.no_grad():
            path = [Image.open(img_path) for img_path in img_paths]
            processed = processor.image_processor.preprocess(
                path,
                return_tensors="pt"
            )
            processed["pixel_values"] = processed["pixel_values"].to(device)
            output = model.vision_model(**processed)
            base_output = basemodel.vision_model(**processed)
            embedding = output.pooler_output.detach().cpu().numpy() - base_output.pooler_output.detach().cpu().numpy()
            return embedding

    embeddings_outpath = construct_output_filename(
        subdir=DATA_DIR / "semantic_clusters",
        prefix=args.prefix,
        suffix="embeddings",
        ext="npy",
    )

    if embeddings_outpath.exists():
        embeddings = np.load(embeddings_outpath)
    else:
        # calculate embeddings for each hashcluster by averaging the embeddings of up to 10 images.
        ind2tpl = []
        embeddings = []
        for tpl in tqdm(tpls):
            tpl = str(tpl)
            ind2tpl.append(tpl)
            file_embeddings = []
            fpaths = hc.filepaths_for_cluster(tpl)
            img_paths = random.sample(fpaths, k=10)
            file_embeddings = embed_image(img_paths)
            cluster_embeddings = np.mean(file_embeddings, axis=0)
            embeddings.append(cluster_embeddings)
        embeddings = np.vstack(embeddings)
        np.save(embeddings_outpath, embeddings)

    ind2tpl_outpath = construct_output_filename(
        subdir=DATA_DIR / "semantic_clusters",
        prefix=args.prefix,
        suffix="index",
        ext="txt",
    )
    if ind2tpl_outpath.exists():
        ind2tpl = ind2tpl_outpath.read_text().split("\n")
    else:
        ind2tpl = [str(tpl) for tpl in tpls]
        with open(ind2tpl_outpath, "w") as f:
            f.write("\n".join(ind2tpl))

    a = (embeddings - embeddings.mean(axis=0, keepdims=True)) / embeddings.std(axis=0, keepdims=True)
    cos_sim = a @ a.T / (np.linalg.norm(a, axis=1) * np.linalg.norm(a, axis=1).reshape(-1, 1))

    _, inds = torch.topk(torch.tensor(cos_sim), 51, dim=0)
    cos_sim = np.zeros_like(cos_sim)
    cos_sim[inds[0], inds] = (0.9 ** np.arange(51)).reshape(-1, 1)

    graph = ig.Graph.Weighted_Adjacency(csr_matrix(cos_sim), mode="max")
    partitions = la.find_partition(graph, la.CPMVertexPartition, weights="weight", resolution_parameter=0.1)




    outpath = construct_output_filename(
        subdir=DATA_DIR / "semantic_clusters",
        prefix=args.prefix,
        suffix="clusters-norm",
        ext="tsv",
    )
    with open(outpath, "w", encoding="utf-8") as f:
        for ind, cluster in enumerate(partitions):
            for tpl_ind in cluster:
                tpl = ind2tpl[tpl_ind]
                f.write(f"{tpl}\t{ind}\n")
    print(len(partitions), "clusters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="all-8-processed-clipdiff")
    main(parser.parse_args())
