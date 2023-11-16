"""Prepares huggingface dataset."""

from collections import Counter
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

from memes.utils import DATA_DIR



dataset = load_dataset("csv", data_files=str(DATA_DIR / "representations/all-8-processed-filtered.csv"))
dataset = dataset["train"]

counts = Counter(dataset["meme_template"])
keeplist = {k for k, v in counts.items() if v >= 100}

def split_ocr(row):
    ocr = eval(row["ocr_output"])
    row["text"] = "</s></s>".join([b[1] for b in ocr])
    return row

dataset = dataset.filter(lambda x: x["meme_template"] in keeplist)
dataset = dataset.map(split_ocr, num_proc=8)

dataset = dataset.class_encode_column("meme_template")

print("Creating train val splits")
unique_text = dataset.unique("text")
train, val = train_test_split(unique_text, test_size=0.1, random_state=0xB1AB)
train = set(train)
val = set(val)

train_subset = dataset.filter(lambda x: x["text"] in train)
val_subset = dataset.filter(lambda x: x["text"] in val)

splits = DatasetDict({"train": train_subset, "test": val_subset})

# splits = dataset.train_test_split(test_size=0.05, stratify_by_column="meme_template")
splits.save_to_disk(DATA_DIR / "representations/all-8-processed-dataset-final")
