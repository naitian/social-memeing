from accelerate import Accelerator
import argparse
from datetime import datetime
import multiprocessing
import os

import evaluate
from datasets import load_from_disk, Image
from transformers import (CLIPImageProcessor, CLIPTokenizer, CLIPModel, DataCollatorWithPadding, AutoConfig, get_scheduler)
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode

from tqdm.auto import tqdm
import wandb


from memes.utils import DATA_DIR, assert_dir, construct_output_filename



def main(args):
    TRAIN_BATCH_SIZE = 512
    EVAL_BATCH_SIZE = 512
    EVAL_STEPS = 5 # 1000 if not args.debug else 10
    LOG_STEPS = 1 # 500 if not args.debug else 10
    BASE_LR = 4e-6
    CLASSIFIER_LR = 4e-6  # following the fine-tuning blog post
    NUM_EPOCHS = 3
    FROM_SCRATCH = False
    MIXED_PRECISION = True

    precision = "bf16" if MIXED_PRECISION else "no"
    accelerator = Accelerator(log_with="wandb", mixed_precision=precision)

    time = f"-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    debug = "-debug" if args.debug else ""
    scratch = "-scratch" if FROM_SCRATCH else "-pretrain"
    mixed = "-bf16" if MIXED_PRECISION else ""
    run_name = f"clip{scratch}{mixed}{time}{debug}"
    accelerator.init_trackers("latent-memes", init_kwargs={"wandb": {"name":run_name}})

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def tokenize(batch):
        # set max length to 77 to match clip-vit-base-patch32 defaults
        # should not be a big deal bc we have pretty short strings
        return tokenizer(batch["text"], truncation=True, max_length=77, padding="max_length")


    def process_image(batch):
        image_paths = batch["resampled_img_path"]
        processed = image_processor(images=image_paths, return_tensors="pt", padding=True)
        batch["pixel_values"] = processed.pixel_values
        return batch

    dataset = load_from_disk(DATA_DIR / "representations/all-8-processed-clip").cast_column("resampled_img_path", Image())
    dataset = dataset.remove_columns(['post_id', 'subreddit', 'meme_hash', 'ocr_output', 'img_path', 'meme_template'])
    # num_labels = dataset["train"].features["labels"].num_classes

    if args.debug:
        train_dataset = dataset["train"].select(range(1000)).map(tokenize, batched=True, num_proc=multiprocessing.cpu_count())
        eval_dataset = dataset["test"].select(range(100)).map(tokenize, batched=True, num_proc=multiprocessing.cpu_count())
    else:
        train_dataset = dataset["train"].map(
            tokenize,
            batched=True,
            num_proc=multiprocessing.cpu_count()
        )
        eval_dataset = dataset["test"].map(
            tokenize,
            batched=True,
            num_proc=multiprocessing.cpu_count()
        ).select(range(1000))  # we just take 1000 because we're running into OOM issues

    def collator(batch):
        pixel_values = torch.stack([d["pixel_values"] for d in batch])
        input_ids = torch.tensor([d["input_ids"] for d in batch], dtype=torch.long)
        attention_mask = torch.tensor([d["attention_mask"] for d in batch], dtype=torch.long)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_loss": True,
        }

    train_dataset.set_format("torch")
    train_dataset = train_dataset.remove_columns("text")
    eval_dataset.set_format("torch")
    eval_dataset = eval_dataset.remove_columns("text")

    train_dataset.set_transform(process_image, output_all_columns=True)
    eval_dataset.set_transform(process_image, output_all_columns=True)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=TRAIN_BATCH_SIZE, collate_fn=collator)
    eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=EVAL_BATCH_SIZE, collate_fn=collator)

    if not FROM_SCRATCH:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise NotImplementedError()
        # config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")
        # model = CLIPModel.from_config(config)
    # model.to(device)

    optimizer = AdamW(model.parameters(), lr=BASE_LR)

    num_epochs = NUM_EPOCHS
    # num_training_steps = num_epochs * len(train_dataloader)
    # lr_scheduler = get_scheduler(
    #     name="constant",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps
    # )

    metric = evaluate.load("accuracy")


    output_dir = DATA_DIR / "representations/clip/checkpoints" / run_name
    assert_dir(output_dir)

    accelerator.print(output_dir)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    # accelerator.register_for_checkpointing(lr_scheduler)

    @accelerator.on_main_process
    def _save_model(model, epoch, step, best=False):
        suffix = "best" if best else f"checkpoint-{epoch}_{step}"
        model.module.save_pretrained(
            output_dir / suffix,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )

    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    step = 0
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # batch = {k: (v.to(device) if k != "return_loss" else v) for k, v in batch.items()}
            # import pdb; pdb.set_trace()
            outputs = model(**batch)
            loss = outputs.loss
            # print(loss)
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            # TODO: modify the accuracy metric for contrastive context!

            if step % EVAL_STEPS == 0 and step > 0:
                model.eval()
                for batch in eval_dataloader:
                    # batch = {k: (v.to(device) if k != "return_loss" else v) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)
                    logits = outputs.logits_per_image
                    predictions = torch.argmax(logits, dim=-1)
                    metric.add_batch(predictions=predictions, references=torch.tensor(range(predictions.shape[0])))
                accuracy = metric.compute()
                if outputs.loss < best_val_loss:
                    _save_model(model, epoch, step, best=True)
                accelerator.log({
                    "eval/epoch": epoch,
                    "eval/loss": outputs.loss,
                    "eval/accuracy": accuracy
                })

                model.train()

            if step % LOG_STEPS == 0:
                logits = outputs.logits_per_image
                predictions = torch.argmax(logits, dim=-1).detach()
                accuracy = metric.compute(predictions=predictions, references=torch.tensor(range(predictions.shape[0])))
                # base_lr = lr_scheduler.get_last_lr()
                base_lr = BASE_LR
                accelerator.log({
                    "train/epoch": epoch,
                    "train/loss": loss,
                    "train/accuracy": accuracy,
                    "train/lr": base_lr,
                })

            step += 1
        accelerator.print("Saving checkpoint")
        _save_model(model, epoch, step)
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    main(parser.parse_args())
