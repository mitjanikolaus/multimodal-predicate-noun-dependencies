import argparse
import json
import os

import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

from src.models import CLIPModelRepresentation, UNITERModelRepresentation, \
    VILTModelRepresentation, LXMERTModelRepresentation, \
    VINVLModelRepresentation, VisualBERTModelRepresentation, VILBERTModelRepresentation, OscarModelRepresentation

from src.utils import get_local_image_path, transform_to_per_pair_results, load_visual_feats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval_2afc(model, args):
    """Evaluate model using 2-alternative forced choice task."""

    samples = json.load(open(args.eval_set, "rb"))

    results = []

    for sample in tqdm(samples):
        if args.alt_sentences:
            text_target = sample["sentence_target_alt"]
            text_distractor = sample["sentence_distractor_alt"]
        else:
            text_target = sample["sentence_target"]
            text_distractor = sample["sentence_distractor"]

        path_example_image = get_local_image_path(sample["img_filename"], args.images_dir, sample["relationship_target"], args.cropped)

        prob_target_match, prob_distractor_match = model.get_img_sent_match_scores(text_target, text_distractor, path_example_image)
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
            "model": args.model,
        })
        results.append(sample)

        # show_sample(sample, args.images_dir, print_bounding_boxes=False)

    results = pd.DataFrame(results)
    file_name = "results"
    if args.cropped:
        file_name += "_cropped_images"
    if args.alt_sentences:
        file_name += "_alt_sentences"
    results_dir = f"runs/sentence-semantics/{args.model}"
    os.makedirs(results_dir, exist_ok=True)
    results.to_csv(os.path.join(results_dir, file_name+".csv"), index=False)

    results_pairs = transform_to_per_pair_results(results)
    print(f"Accuracy: {round(100*results_pairs.result.mean(), 2)}%")


def get_model(model, device, visual_feats_file='', offline=False, for_pretraining=False):
    if model == 'LXMERT':
        rep_model = LXMERTModelRepresentation(device, visual_feats_file, for_pretraining)
    elif model == 'VisualBERT':
        rep_model = VisualBERTModelRepresentation(device, visual_feats_file)
    elif model == 'UNITER':
        rep_model = UNITERModelRepresentation(device, visual_feats_file, offline=offline)
    elif model == 'VILT':
        rep_model = VILTModelRepresentation(device, offline=offline)
    elif model == 'CLIP':
        rep_model = CLIPModelRepresentation(device)
    elif model == 'Oscar':
        rep_model = OscarModelRepresentation(device, visual_feats_file)
    elif model == 'VINVL':
        rep_model = VINVLModelRepresentation(device, visual_feats_file)
    elif model == 'VILBERT':
        rep_model = VILBERTModelRepresentation(device, visual_feats_file)
    else:
        raise RuntimeError("Unknown model: ", model)
    return rep_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=['VILT', 'UNITER', 'LXMERT', 'VisualBERT', 'CLIP', 'Oscar', 'VINVL', 'VILBERT'])
    parser.add_argument("--eval-set", type=str, default="data/sentence-semantics/eval_set.json")
    parser.add_argument("--img-features-path", type=str)

    parser.add_argument("--images-dir", type=str, default=os.path.expanduser("~/data/multimodal_evaluation/images/"),
                        help="Local dir with images")

    parser.add_argument("--cropped", action="store_true", help="Evaluate using cropped images (sanity check)")

    parser.add_argument("--alt-sentences", action="store_true", help="Evaluate using alternative sentences")

    parser.add_argument("--offline", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    model = get_model(args.model, device, args.img_features_path, args.offline, for_pretraining=True)

    eval_2afc(model, args)
