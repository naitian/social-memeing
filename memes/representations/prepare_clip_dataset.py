"""Prepares huggingface dataset."""

from collections import Counter, defaultdict
from pathlib import Path
import random
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm.auto import tqdm

from memes.utils import DATA_DIR


dataset = load_dataset(
    "csv", data_files=str(DATA_DIR / "representations/all-8-processed-filtered.csv")
)
dataset = dataset["train"]


def get_template_group(dataset):
    templates = dataset["meme_template"]
    img_paths = dataset["img_path"]
    template_group = defaultdict(set)
    for i, tpl in enumerate(tqdm(templates)):
        if not Path(img_paths[i]).exists():
            continue
        template_group[tpl].add(img_paths[i])
    return template_group

template_group = get_template_group(dataset)
keeplist = {k for k, v in template_group.items() if len(v) >= 100}

def split_ocr(row):
    ocr = eval(row["ocr_output"])
    # I use <|endoftext|> here because it looks like that token is used for all
    # sorts of purposes in CLIP including unk and pad (see
    # https://huggingface.co/transformers/model_doc/clip.html#transformers.CLIPTokenizer)
    row["text"] = "<|endoftext|>".join([b[1] for b in ocr])
    return row




def filter_corrupt_images(examples):
    """remove problematic images"""
    valid_images = []
    for image_file in examples["resampled_img_path"]:
        try:
            # assert Path(image_file).exists()
            Image.open(image_file)
            valid_images.append(True)
        except Exception:
            valid_images.append(False)
    return valid_images

def resample_image_path(examples):
    output = []
    for tpl in examples["meme_template"]:
        output.append(random.choice(tuple(template_group[tpl])))
    return {"resampled_img_path": output}

print("Filtering low count templates")
dataset = dataset.filter(lambda x: x["meme_template"] in keeplist)
print("Resampling image paths")
dataset = dataset.map(resample_image_path, batched=True, num_proc=8)
# Skip filtering bc I'm pretty sure we would've caught any corrupt images much
# earlier in the pipeline (e.g. during phashing)
# print("Filtering corrupt images")
# dataset = dataset.filter(
#     filter_corrupt_images, batched=True, num_proc=16
# )
print("Splitting OCR")
dataset = dataset.map(split_ocr, num_proc=8)

print("Creating train val splits")
unique_text = dataset.unique("text")
train, val = train_test_split(unique_text, test_size=0.1, random_state=0xB1AB)
train = set(train)
val = set(val)

train_subset = dataset.filter(lambda x: x["text"] in train)
val_subset = dataset.filter(lambda x: x["text"] in val)

splits = DatasetDict({"train": train_subset, "test": val_subset})

dataset = dataset.class_encode_column("meme_template")

# splits = dataset.train_test_split(test_size=0.05, stratify_by_column="meme_template")
splits.save_to_disk(DATA_DIR / "representations/all-8-processed-clip-final")
