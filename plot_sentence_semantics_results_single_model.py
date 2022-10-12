import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from generate_results_tables import get_bootstrapped_scores
from src.utils import transform_to_per_pair_results, NOUNS

MIN_NUM_TEST_SAMPLES = 5
N_BOOTSTRAP_SAMPLES = 100


def bootstrap_scores(results):
    results_boot = []
    for value in results.concept.unique():
        results_data_word = results[results.concept == value].result
        result_bootstrapped = get_bootstrapped_scores(results_data_word.values, N_BOOTSTRAP_SAMPLES)
        items = [{"result": res, "concept": value} for res in result_bootstrapped]
        results_boot.extend(items)

    return pd.DataFrame.from_records(results_boot)


def multiply_df_for_per_concept_analyses(results):
    results_words_1 = results.copy()
    results_words_1["concept"] = results_words_1["subject"]
    results_words_2 = results.copy()
    results_words_2["concept"] = results_words_2["object"]
    results_words_3 = results.copy()
    results_words_3["concept"] = results_words_3["word_distractor"]

    results_words = pd.concat([results_words_1, results_words_2, results_words_3], ignore_index=True)

    return results_words


def make_plots(args):
    print("Loading: ", args.input_file)
    results = pd.read_csv(args.input_file, index_col=False)

    results_pairs = transform_to_per_pair_results(results)
    print(f"Overall accuracy: {100*results_pairs['result'].mean():.2f}")

    results_concepts = multiply_df_for_per_concept_analyses(results_pairs)

    words_enough_samples = [k for k, v in results_concepts.groupby("concept").size().to_dict().items() if v >= MIN_NUM_TEST_SAMPLES]
    results_concepts = results_concepts[results_concepts.concept.isin(words_enough_samples)]
    data_bootstrapped = bootstrap_scores(results_concepts)

    _, axes = plt.subplots(2, 1, figsize=(4, 5), sharex="none", gridspec_kw={'height_ratios': [1, 5]})

    data_nouns = data_bootstrapped[data_bootstrapped.concept.isin(NOUNS)]
    sorted_words = data_nouns.groupby("concept").agg({"result": "mean"}).sort_values(by="result").index
    sns.pointplot(data=data_nouns, y="concept", x="result", color="black", ci="sd", errwidth=1, join=False,
                  order=sorted_words, ax=axes[0])
    axes[0].set_title("Nouns")
    axes[0].set_xticks([0, 0.25, 0.5, 0.75, 1])
    axes[0].set_xticklabels([])
    axes[0].xaxis.label.set_visible(False)
    axes[0].yaxis.label.set_visible(False)
    axes[0].axvline(x=0.25, color="black", linestyle='--')

    plt.tight_layout()

    data_predicates = data_bootstrapped[~data_bootstrapped.concept.isin(NOUNS)].copy()

    full_predicates = {}
    predicates = data_predicates.concept.unique()
    for predicate in predicates:
        full_pred = results[results.object == predicate].iloc[0].sentence_target.split("is ")[1]
        full_predicates[predicate] = full_pred

    data_predicates.concept.replace(full_predicates, inplace=True)
    sorted_words = data_predicates.groupby("concept").agg({"result": "mean"}).sort_values(by="result").index
    sns.pointplot(data=data_predicates, y="concept", x="result", color="black", ci="sd", errwidth=1, join=False,
                  order=sorted_words, ax=axes[1])
    axes[1].set_title("Predicates")
    plt.xlabel("Accuracy")
    plt.ylabel("")
    axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
    axes[1].set_xticklabels([0, 0.25, 0.5, 0.75, 1])
    axes[1].axvline(x=0.25, color="black", linestyle='--')
    axes[1].text(x=0.18, y=19, s="Chance", rotation="vertical")
    plt.tight_layout()

    plt.savefig(os.path.join(os.path.dirname(args.input_file), f"accuracies_per_concept.pdf"), dpi=300)

    plt.show()


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input-file", type=str, required=True)

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    make_plots(args)
