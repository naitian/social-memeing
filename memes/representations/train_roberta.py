import argparse
from datetime import datetime
import multiprocessing
import os
from accelerate import Accelerator

import evaluate
from datasets import load_from_disk
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, AutoConfig,
                          Trainer, TrainingArguments, get_scheduler, DataCollatorWithPadding, create_optimizer)
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import wandb


from memes.utils import DATA_DIR, assert_dir, construct_output_filename

def main(args):
    TRAIN_BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 32
    EVAL_STEPS = 1000 if not args.debug else 10
    LOG_STEPS = 500 if not args.debug else 10
    BASE_LR = 1e-6
    CLASSIFIER_LR = 1e-5
    NUM_EPOCHS = 3
    FROM_SCRATCH = False
    MIXED_PRECISION = False

    time = f"-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    debug = "-debug" if args.debug else ""
    scratch = "-scratch" if FROM_SCRATCH else "-pretrain"
    run_name = f"c-roberta_pt_lowlr{scratch}{time}{debug}"

    precision = "bf16" if MIXED_PRECISION else "no"
    accelerator = Accelerator(log_with="wandb", mixed_precision=precision)

    accelerator.init_trackers("latent-memes", init_kwargs={"wandb": {"name":run_name}})
    # wandb.init(
    #     project="latent-memes",
    #     name=run_name,
    # )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    # training_args = TrainingArguments(
    #     output_dir=DATA_DIR / f"representations/classifier/checkpoints/roberta-{time}",
    #     evaluation_strategy="steps",
    #     eval_steps=500 if not args.debug else 100,
    #     logging_steps=100 if not args.debug else 5,
    #     report_to="wandb",
    #     run_name=f"c-roberta-{time}{debug}",
    #     auto_find_batch_size=True,
    #     save_steps=0.1,
    #     learning_rate=1e-3,
    #     lr_scheduler_type="linear",
    #     warmup_steps=2000,
    # )

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    dataset = load_from_disk(DATA_DIR / "representations/all-8-processed-dataset")
    dataset = dataset.remove_columns(['post_id', 'subreddit', 'meme_hash', 'ocr_output', 'img_path'])
    dataset = dataset.rename_column("meme_template", "labels")
    num_labels = dataset["train"].features["labels"].num_classes

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


    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset.set_format("torch")
    train_dataset = train_dataset.remove_columns("text")
    eval_dataset.set_format("torch")
    eval_dataset = eval_dataset.remove_columns("text")

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=TRAIN_BATCH_SIZE, collate_fn=collator)
    eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=EVAL_BATCH_SIZE, collate_fn=collator)

    if not FROM_SCRATCH:
        model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)
    else:
        config = AutoConfig.from_pretrained("roberta-base", num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_config(config)
    model.to(device)

    optimizer = AdamW([
        {"params": model.roberta.parameters(), "lr": BASE_LR},
        {"params": model.classifier.parameters(), "lr": CLASSIFIER_LR}
    ])

    num_epochs = NUM_EPOCHS
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    metric = evaluate.load("accuracy")


    output_dir = DATA_DIR / "representations/classifier/checkpoints" / run_name
    assert_dir(output_dir)

    @accelerator.on_main_process
    def _save_model(model, epoch, step, best=False):
        suffix = "best" if best else f"checkpoint-{epoch}_{step}"
        model_path = construct_output_filename(
            subdir=output_dir,
            prefix=None,
            suffix=suffix,
            ext="pt",
        )
        torch.save(model.state_dict(), model_path)

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    progress_bar = tqdm(range(num_training_steps))
    model.train()
    step = 0
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # import pdb; pdb.set_trace()
            loss = outputs.loss
            # print(loss)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            if step % EVAL_STEPS == 0 and step > 0:
                model.eval()
                for batch in eval_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    metric.add_batch(predictions=predictions, references=batch["labels"])
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
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1).detach()
                accuracy = metric.compute(references=batch["labels"], predictions=predictions)
                base_lr, classifier_lr = lr_scheduler.get_last_lr()
                accelerator.log({
                    "train/epoch": epoch,
                    "train/loss": loss,
                    "train/accuracy": accuracy,
                    "train/base_lr": base_lr,
                    "train/classifier_lr": classifier_lr,
                })

            step += 1
        accelerator.print("Saving checkpoint")
        _save_model(model, epoch, step)
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    main(parser.parse_args())