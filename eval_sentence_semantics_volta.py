import os
import json
import random
import argparse
from io import open

import pandas as pd
import numpy as np

import torch
from tqdm import tqdm

from transformers import AutoTokenizer
import torch.nn.functional as F

from src.utils import load_visual_feats, transform_sample
from src.utils import get_local_image_path, transform_to_per_pair_results
from volta.config import BertConfig
from volta.encoders import BertForVLPreTraining


def calc_img_sent_matching_score(test_sentence, sample, visual_feat_dict, model, tokenizer, config, args):
    path_example_image = get_local_image_path(sample["img_filename"], args.images_dir, sample["relationship_target"],
                                              args.cropped)
    visual_features_key = os.path.basename(path_example_image)
    visual_feat_raw = visual_feat_dict[visual_features_key]

    input_ids, input_mask, segment_ids, image_feat, image_loc, image_masks = transform_sample(tokenizer, test_sentence,
                                                                                             visual_feat_raw,
                                                                                             num_locs=config.num_locs,
                                                                                             max_seq_length=args.max_seq_length,
                                                                                             add_global_imgfeat=config.add_global_imgfeat)

    # decoded_sequence = tokenizer.decode(input_ids[0])
    # print(decoded_sequence)

    with torch.no_grad():
        _, _, seq_relationship_score, _, _, _ = model(input_ids, image_feat, image_loc, segment_ids, input_mask,
                                                      image_masks)

    # Apply softmax
    softmaxed = F.softmax(seq_relationship_score[0], dim=0)
    # return the probability for image-sentence match:
    return softmaxed[0].item()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval-set", type=str, default="data/sentence-semantics/eval_set.json")
    parser.add_argument("--img-features-path", type=str)
    parser.add_argument("--images-dir", type=str, default=os.path.expanduser("~/data/multimodal_evaluation/"),
                        help="Local dir with images")
    parser.add_argument("--cropped", action="store_true", help="Evaluate using cropped images (sanity check)")


    # Model
    parser.add_argument("--from_pretrained", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, roberta-base, ...")
    parser.add_argument("--config_file", type=str, default="config/vilbert_base.json",
                        help="The config file which specified the model details.")

    # Output
    parser.add_argument("--output_dir", default="checkpoints", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--logdir", default="logs", type=str,
                        help="The logging directory where the training logs will be written.")
    # Text
    parser.add_argument("--max_seq_length", default=36, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load config
    config = BertConfig.from_json_file(args.config_file)

    # Output dirs
    timestamp = args.config_file.split("/")[1].split(".")[0]
    save_path = os.path.join(args.output_dir, timestamp)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        # save all the hidden parameters.
        with open(os.path.join(save_path, "command.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)

    # Datasets
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)

    # Model
    model = BertForVLPreTraining.from_pretrained(args.from_pretrained, config=config, default_gpu=True)#, from_hf=True)

    model_name = args.config_file.split("/")[-1].split(".")[0].upper()

    # Move to GPU(s)
    if not device.type == "cpu":
        model.cuda()

    # Data
    samples = json.load(open(args.eval_set, "rb"))
    visual_feat_dict = load_visual_feats(args.img_features_path)

    # Eval
    model.eval()

    results = []
    for sample in tqdm(samples):
        text_target = sample["sentence_target"]
        text_distractor = sample["sentence_distractor"]
        prob_target_match = calc_img_sent_matching_score(text_target, sample, visual_feat_dict, model, tokenizer, config, args)
        prob_distractor_match = calc_img_sent_matching_score(text_distractor, sample, visual_feat_dict, model, tokenizer, config, args)
        result_normal = bool(prob_target_match > prob_distractor_match)
        result_separate = np.mean([bool(prob_target_match >= 0.5), bool(prob_distractor_match < 0.5)])
        result_strict = bool(prob_target_match >= 0.5) and bool(prob_distractor_match < 0.5)

        sample.update({
            "result": result_normal,
            "result_separate": result_separate,
            "result_strict": result_strict,
            "prob_target_match": prob_target_match,
            "prob_distractor_match": prob_distractor_match,
            "cropped_image": args.cropped,
            "model": model_name,
        })
        results.append(sample)

    results = pd.DataFrame(results)
    file_name = "results"
    if args.cropped:
        file_name += "_cropped_images"
    results_dir = f"runs/sentence-semantics/{model_name}"
    os.makedirs(results_dir, exist_ok=True)
    results.to_csv(os.path.join(results_dir, file_name + ".csv"), index=False)

    results_pairs = transform_to_per_pair_results(results)
    print(f"Accuracy: {round(100*results_pairs.result.mean(), 2)}%")


if __name__ == "__main__":
    main()
