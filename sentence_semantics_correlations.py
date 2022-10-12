import argparse
import itertools
import json
import os.path
from ast import literal_eval
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils import bb_size, SUBJECT, OBJECT, SYNONYMS, transform_label, load_visual_feats

ROUND_DECIMALS = 2


def bb_distance_from_center(rel):
    bb = rel["bounding_box"]
    center_x = bb[0] + (bb[2] / 2)
    dist_x = abs(0.5 - center_x)
    center_y = bb[1] + (bb[3] / 2)
    dist_y = abs(0.5 - center_y)

    return np.sqrt(dist_x**2 + dist_y**2)


def tuple_to_str(tuple):
    return tuple[0] + "_" + tuple[1]


def load_conceptual_captions_stats(subjects, objects):
    # Training data downloaded from https://ai.google.com/research/ConceptualCaptions/download
    data = pd.read_csv(
        "data/conceptual_captions/Train_GCC-training.tsv", sep="\t", header=None
    )
    captions = data[0].values

    captions_words = [caption.lower().split(" ") for caption in captions]

    # Freqs for words
    words = [transform_label(word) if len(transform_label(word).split(" ")) == 1 else transform_label(word).split(" ")[1] for word in set(list(subjects) + list(objects))]
    word_freqs = Counter(itertools.chain(*captions_words))
    word_log_freqs = {w: np.log(f) for w, f in word_freqs.items() if w in words}
    json.dump(word_log_freqs, open("data/conceptual_captions/word_occ_stats.json", "w"))

    # Freqs for tuples
    tuples = list(zip(subjects, objects))
    tuple_freqs = {t: 0 for t in set(tuples)}
    for s, o in set(tuples):
        for caption_words in captions_words:
            words = []
            for word in [s, o]:
                word = transform_label(word)
                if len(word.split(" ")) > 1:
                    words.append(word.split(" ")[1])
                else:
                    words.append(word)
            is_match = True
            for word in words:
                if word.lower() not in caption_words:
                    is_match = False
                    break
            if is_match:
                tuple_freqs[s, o] += 1

    tuple_log_freqs = {tuple_to_str(p): np.log(f) for p, f in tuple_freqs.items()}
    json.dump(tuple_log_freqs, open("data/conceptual_captions/tuple_occ_stats.json", "w"))


def get_tuple_log_freq(row, log_freqs, target_rel):
    if target_rel:
        subj = row.relationship_target[SUBJECT]
        obj = row.relationship_target[OBJECT]
    else:
        if row.pos == "subject":
            subj = row.relationship_distractor[SUBJECT]
            obj = row.relationship_target[OBJECT]
        elif row.pos == "object":
            subj = row.relationship_target[SUBJECT]
            obj = row.relationship_distractor[OBJECT]
        else:
            raise NotImplementedError("POS: ", row.pos)
    freq = log_freqs[tuple_to_str((subj, obj))]
    return np.round(freq, ROUND_DECIMALS)


def get_word_log_freq(word, log_freqs):
    if len(transform_label(word).split(" ")) == 1:
        return log_freqs[transform_label(word)]
    else:
        return log_freqs[transform_label(word).split(" ")[1]]


def get_sentence_perplexity(sentence, sentence_perplexities):
    return sentence_perplexities[sentence]


def normalize(data, column_name):
    min, max, mean = data[column_name].min(), data[column_name].max(), data[column_name].mean()
    data[column_name] = (data[column_name] - mean) / (max - min) * (1 - 0)


def replace_synonyms_in_rel(rel):
    rel[SUBJECT] = SYNONYMS[rel[SUBJECT]][0]
    rel[OBJECT] = SYNONYMS[rel[OBJECT]][0]
    return rel


def replace_synonyms(data):
    data["subject"] = data.subject.apply(lambda x: SYNONYMS[x][0])
    data["object"] = data.object.apply(lambda x: SYNONYMS[x][0])

    data["relationship_target"] = data.relationship_target.apply(replace_synonyms_in_rel)
    data["relationship_distractor"] = data.relationship_distractor.apply(replace_synonyms_in_rel)


def transform_visual_genome_labels(labels):
    transformed_labels = []
    for label in labels:
        if label in ["sunglasses", "eyeglasses", "eye glasses"]:
            transformed_labels.append("glasses")
        elif label in ["wine glass"]:
            transformed_labels.append("glass")
        elif label in ["water bottle", "beer bottle", "wine bottle"]:
            transformed_labels.append("bottle")
        elif label in ["cell phone", "iphone", "cellphone", "smartphone"]:
            transformed_labels.append("phone")
        elif label in ["bicycle", "dirt bike"]:
            transformed_labels.append("bike")
        elif label in ["cowboy hat"]:
            transformed_labels.append("hat")
        elif label in ["lying", "laying down"]:
            transformed_labels.append("laying")
        else:
            transformed_labels.append(label)
    return transformed_labels


def get_confidence_score(label, vis_feats):
    CONCEPTS_NOT_IN_VISUAL_GENOME = ["cry", "sing", "guitar", "cello", "drum"]
    if label.lower() in CONCEPTS_NOT_IN_VISUAL_GENOME:
        return None
    else:
        label = transform_label(label)
        label = label.lower()

        if label == "wine glass":
            label = "glass"
        elif label == "mobile phone":
            label = "phone"

        labels = transform_visual_genome_labels(vis_feats["object_names"]) + vis_feats["attrs_names"]
        confidences = list(vis_feats["objects_conf"]) + list(vis_feats["attrs_conf"])
        distractors = [(name, conf) for name, conf in zip(labels, confidences) if label == name]
        max_confidence = max([conf for name, conf in distractors]) if len(distractors) > 0 else 0

    return max_confidence


def get_target_distractor_confidence_diff(results, visual_feat_dict):
    values = []
    for id, row in results.iterrows():
        vis_feats = visual_feat_dict[row["img_filename"]]

        conf_distractor = get_confidence_score(row["word_distractor"], vis_feats)
        conf_target = get_confidence_score(row["word_target"], vis_feats)

        if conf_distractor is None or conf_target is None:
            values.append(None)
        else:
            values.append(conf_target - conf_distractor)

    return values


def calc_correlations(args):
    bottom_up_features_path = os.path.expanduser("~/data/multimodal_evaluation/image_features_2048/img_features_2048_10_100.tsv")
    bottom_up_features = load_visual_feats(bottom_up_features_path)

    correlations = []
    for model in args.models:
        print(model)
        corrs = {"Model": model}

        results = pd.read_csv(f"runs/sentence-semantics/{model}/results.csv", index_col=False, converters={'relationship_target': literal_eval, 'relationship_distractor': literal_eval})

        results["bb_size_target"] = [bb_size(rel) for rel in results.relationship_target]
        results["bb_size_distractor"] = [bb_size(rel) for rel in results.relationship_distractor]
        results["bb_size_diff"] = round(results["bb_size_target"] - results["bb_size_distractor"], ROUND_DECIMALS)

        results["bb_dist_center_target"] = results.relationship_target.apply(bb_distance_from_center)
        results["bb_dist_center_distractor"] = results.relationship_distractor.apply(bb_distance_from_center)
        results["bb_dist_center_diff"] = round(results["bb_dist_center_target"] - results["bb_dist_center_distractor"], ROUND_DECIMALS)

        replace_synonyms(results)

        results["confidence_diff"] = get_target_distractor_confidence_diff(results, bottom_up_features)

        if not os.path.isfile("data/conceptual_captions/word_occ_stats.json"):
            load_conceptual_captions_stats(results.subject.values, results.object.values)
        word_log_freqs = json.load(open("data/conceptual_captions/word_occ_stats.json"))
        tuple_log_freqs = json.load(open("data/conceptual_captions/tuple_occ_stats.json"))

        results["log_freq_target_tuple"] = results.apply(get_tuple_log_freq, log_freqs=tuple_log_freqs, target_rel=True, axis=1)
        results["log_freq_distractor_tuple"] = results.apply(get_tuple_log_freq, log_freqs=tuple_log_freqs, target_rel=False, axis=1)
        results["log_freq_tuple_diff"] = round(results["log_freq_target_tuple"] - results["log_freq_distractor_tuple"], ROUND_DECIMALS)

        results["log_freq_target_word"] = results.word_target.apply(get_word_log_freq, log_freqs=word_log_freqs)
        results["log_freq_distractor_word"] = results.word_distractor.apply(get_word_log_freq, log_freqs=word_log_freqs)
        results["log_freq_word_diff"] = round(results["log_freq_target_word"] - results["log_freq_distractor_word"],
                                         ROUND_DECIMALS)

        sentence_perplexities = json.load(open("data/conceptual_captions/sentence_perplexities.json"))
        results["ppl_target_sentence"] = results.sentence_target.apply(get_sentence_perplexity, sentence_perplexities=sentence_perplexities)
        results["ppl_distractor_sentence"] = results.sentence_distractor.apply(get_sentence_perplexity, sentence_perplexities=sentence_perplexities)
        results["ppl_diff"] = round(results["ppl_target_sentence"] - results["ppl_distractor_sentence"], ROUND_DECIMALS)

        results["prob_diff"] = round(results["prob_target_match"] - results["prob_distractor_match"], ROUND_DECIMALS)

        pearson_r = pearsonr(results.bb_size_diff, results.prob_diff)
        print("bb_size_diff pearson r", pearson_r)
        corrs["bb_size_diff"] = f"{pearson_r[0]:.2f} (p={pearson_r[1]:.2f})"
        # plt.title(f"bb_size_diff\nPearson r: {pearson_r}")
        # sns.scatterplot(data=results, x="bb_size_diff", y="prob_diff")
        # plt.tight_layout()
        #
        # plt.figure()
        pearson_r = pearsonr(results.bb_dist_center_diff, results.prob_diff)
        print("bb_dist_center_diff pearson r", pearson_r)
        corrs["bb_dist_center_diff"] = f"{pearson_r[0]:.2f} (p={pearson_r[1]:.2f})"
        # plt.title(f"bb_dist_center_diff\nPearson r: {pearson_r}")
        # sns.scatterplot(data=results, x="bb_dist_center_diff", y="prob_diff")
        # plt.tight_layout()
        #
        # plt.figure()
        # pearson_r = pearsonr(results.log_freq_word_diff, results.prob_diff)
        # print("log_freq_word_diff pearson r", pearson_r)
        # corrs["log_freq_word_diff"] = f"{pearson_r[0]:.2f} (p={pearson_r[1]:.2f})"

        # plt.title(f"log_freq_word_diff\nPearson r: {pearson_r}")
        # sns.scatterplot(data=results, x="log_freq_word_diff", y="prob_diff")
        # plt.tight_layout()
        #
        # plt.figure()
        # pearson_r = pearsonr(results.log_freq_tuple_diff, results.prob_diff)
        # print("log_freq_tuple_diff pearson r", pearson_r)
        # corrs["log_freq_tuple_diff"] = f"{pearson_r[0]:.2f} (p={pearson_r[1]:.2f})"
        # plt.title(f"log_freq_tuple_diff\nPearson r: {pearson_r}")
        # sns.scatterplot(data=results, x="log_freq_tuple_diff", y="prob_diff")
        # plt.tight_layout()
        #
        # plt.show()

        # plt.figure()
        pearson_r = pearsonr(results.ppl_diff, results.prob_diff)
        print("ppl_diff pearson r", pearson_r)
        corrs["ppl_diff"] = f"{pearson_r[0]:.2f} (p={pearson_r[1]:.2f})"
        # plt.title(f"ppl_diff\nPearson r: {pearson_r}")
        # sns.scatterplot(data=results, x="ppl_diff", y="prob_diff")
        # plt.tight_layout()
        #
        # plt.show()

        res_no_nans = results.dropna(subset=["confidence_diff"])
        pearson_r = pearsonr(res_no_nans.confidence_diff, res_no_nans.prob_diff)
        print("confidence_diff pearson r", pearson_r)
        corrs["confidence_diff"] = f"{pearson_r[0]:.2f} (p={pearson_r[1]:.2f})"

        # plt.title(f"confidence_diff\nPearson r: {pearson_r}")
        # sns.scatterplot(data=res_no_nans, x="confidence_diff", y="prob_diff")
        # plt.tight_layout()
        #
        # plt.show()

        correlations.append(corrs)

    correlations = pd.DataFrame(correlations)
    correlations.to_latex("runs/sentence-semantics/correlations.tex", index=False, na_rep="")


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--models", type=str, nargs="+")

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    calc_correlations(args)
