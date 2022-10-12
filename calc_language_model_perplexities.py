import argparse
import itertools
import json
import math
import os.path
from ast import literal_eval
from collections import Counter

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, load_metric
from datasets.utils.file_utils import get_datasets_user_agent


def calc_ppls(args):
    samples = json.load(open(args.eval_set, "rb"))

    sentences = set([s["sentence_target"] for s in samples] + [s["sentence_distractor"] for s in samples])

    checkpoint_path = "test_trainer/checkpoint-622000"
    model = BertForMaskedLM.from_pretrained(checkpoint_path)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model.eval()

    def sentence_ppl(model, tokenizer, sentence):
        tensor_input = tokenizer.encode(sentence, return_tensors='pt')
        input_repeated = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
        masked_input = input_repeated.masked_fill(mask == 1, tokenizer.mask_token_id)
        labels = input_repeated.masked_fill(masked_input != tokenizer.mask_token_id, -100)
        loss = model(masked_input, labels=labels).loss
        return np.exp(loss.item())

    perplexities = {}
    for sentence in sentences:
        ppl = sentence_ppl(sentence=sentence, model=model, tokenizer=tokenizer)
        print(sentence, ppl)
        perplexities[sentence] = ppl

    json.dump(perplexities, open("data/conceptual_captions/sentence_perplexities.json", "w"))


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--eval-set", type=str, default="data/sentence-semantics/eval_set.json")

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    calc_ppls(args)
