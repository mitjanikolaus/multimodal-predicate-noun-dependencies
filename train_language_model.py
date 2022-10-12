import argparse
import itertools
import json
import math
import os.path
from ast import literal_eval
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, load_metric
from datasets.utils.file_utils import get_datasets_user_agent

import torch


BATCH_SIZE = 16

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(args):
    print(device)
    data = load_dataset("conceptual_captions")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    data = data.rename_column("caption", "input_ids")
    data = data.remove_columns("image_url")

    def tokenize_function(examples):
        return tokenizer(examples["input_ids"], padding="max_length", truncation=True)


    tokenized_datasets = data.map(tokenize_function, batched=True, batch_size=BATCH_SIZE)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)

    model.train()

    training_args = TrainingArguments(output_dir="test_trainer_lr_5e-5",
                                      save_steps=1000,
                                      per_device_eval_batch_size=BATCH_SIZE,
                                      per_device_train_batch_size=BATCH_SIZE,
                                      evaluation_strategy="steps",
                                      eval_steps=1000,
                                      learning_rate=5e-5,
                                      warmup_steps=1_000,
                                      weight_decay=0.01,
                                      fp16=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    initial_eval = trainer.evaluate()
    print(initial_eval)
    print(f"Perplexity: {math.exp(initial_eval['eval_loss']):.2f}")

    trainer.train()

    final_eval = trainer.evaluate()
    print(f"Perplexity: {math.exp(final_eval['eval_loss']):.2f}")


def parse_args():
    argparser = argparse.ArgumentParser()

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
