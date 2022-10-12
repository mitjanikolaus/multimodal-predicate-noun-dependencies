import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from generate_results_tables import get_bootstrapped_scores
from src.utils import transform_to_per_pair_results, NOUNS

MIN_NUM_TEST_SAMPLES = 5
N_BOOTSTRAP_SAMPLES = 100


def bootstrap_scores(results, column):
    results_boot = []
    for value in results[column].unique():
        results_data_word = results[results[column] == value].result
        result_bootstrapped = get_bootstrapped_scores(results_data_word.values, N_BOOTSTRAP_SAMPLES)
        items = [{"result": res, column: value} for res in result_bootstrapped]
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
    all_results_bootstrapped_target_distractor = []
    all_results_bootstrapped = []
    for model in args.models:
        print("Model: ", model)
        file_name = f"runs/sentence-semantics/{model}/results.csv"
        results = pd.read_csv(file_name, index_col=False)

        results_pairs = transform_to_per_pair_results(results)
        print(f"Overall accuracy: {100*results_pairs['result'].mean():.2f}")

        results_concepts = multiply_df_for_per_concept_analyses(results_pairs)

        words_enough_samples = [k for k, v in results_concepts.groupby("concept").size().to_dict().items() if v >= MIN_NUM_TEST_SAMPLES]
        results_concepts = results_concepts[results_concepts.concept.isin(words_enough_samples)]
        data_bootstrapped = bootstrap_scores(results_concepts, column="concept")

        data_predicates = data_bootstrapped[~data_bootstrapped.concept.isin(NOUNS)].copy()
        full_predicates = {}
        predicates = data_predicates.concept.unique()
        for predicate in predicates:
            full_pred = results[results.object == predicate].iloc[0].sentence_target.split("is ")[1]
            full_predicates[predicate] = full_pred

        data_predicates.concept.replace(full_predicates, inplace=True)

        data_bootstrapped["model"] = model
        all_results_bootstrapped.append(data_bootstrapped)

        results["target_distractor"] = [(t, d) for t, d in zip(results["word_target"], results["word_distractor"])]
        targets_tuples_enough_samples = [k for k, v in results.groupby("target_distractor").size().to_dict().items() if v >= MIN_NUM_TEST_SAMPLES]
        results = results[results.target_distractor.isin(targets_tuples_enough_samples)]

        data_bootstrapped_target_distractor = bootstrap_scores(results, column="target_distractor")
        data_bootstrapped_target_distractor["model"] = model
        all_results_bootstrapped_target_distractor.append(data_bootstrapped_target_distractor)

    all_results_bootstrapped = pd.concat(all_results_bootstrapped, ignore_index=True)

    _, axes = plt.subplots(2, 1, figsize=(6, 9), sharex="none", gridspec_kw={'height_ratios': [1, 5]})

    data_nouns = all_results_bootstrapped[all_results_bootstrapped.concept.isin(NOUNS)]
    sorted_words = data_nouns.groupby("concept").agg({"result": "mean"}).sort_values(by="result").index
    sns.pointplot(data=data_nouns, y="concept", x="result", hue="model", ci="sd", errwidth=.5, join=False,
                  order=sorted_words, dodge=0.5, scale=0.4, ax=axes[0])
    axes[0].legend_.remove()
    axes[0].set_title("Nouns")
    axes[0].set_xticks([0, 0.25, 0.5, 0.75, 1])
    axes[0].set_xticklabels([])
    axes[0].xaxis.label.set_visible(False)
    axes[0].yaxis.label.set_visible(False)
    axes[0].axvline(x=0.25, color="black", linestyle='--')

    data_predicates = all_results_bootstrapped[~all_results_bootstrapped.concept.isin(NOUNS)].copy()

    sorted_words = data_predicates.groupby("concept").agg({"result": "mean"}).sort_values(by="result").index
    sns.pointplot(data=data_predicates, y="concept", x="result", hue="model", ci="sd", errwidth=.5, join=False,
                  order=sorted_words, dodge=0.5, scale=0.4, ax=axes[1])
    axes[1].set_title("Predicates")
    plt.xlabel("Accuracy")
    plt.ylabel("")
    axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
    axes[1].set_xticklabels([0, 0.25, 0.5, 0.75, 1])
    axes[1].axvline(x=0.25, color="black", linestyle='--')
    axes[1].text(x=0.215, y=21, s="Chance", rotation="vertical")

    plt.tight_layout()

    plt.savefig(os.path.join("runs/sentence-semantics", f"accuracies_per_concept.pdf"), dpi=300)

    all_results_bootstrapped_target_distractor = pd.concat(all_results_bootstrapped_target_distractor, ignore_index=True)
    plt.figure(figsize=(6, 9))
    sns.pointplot(data=all_results_bootstrapped_target_distractor, y="target_distractor", x="result", hue="model", errwidth=0.5, dodge=0.5, scale=0.4, ci="sd", join=False)

    plt.xlabel("Accuracy")
    plt.ylabel("(target, distractor)")

    plt.axvline(x=0.5, color="black", linestyle='--')
    plt.text(x=0.45, y=46, s="Chance", rotation="vertical")

    plt.tight_layout()
    plt.savefig(os.path.join("runs/sentence-semantics", "accuracies_target_distractor.pdf"), dpi=300)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--models", type=str, nargs="+")

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    make_plots(args)
