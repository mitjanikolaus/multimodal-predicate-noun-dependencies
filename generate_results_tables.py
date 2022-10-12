import argparse
import os.path

import pandas as pd
import numpy as np

from src.utils import transform_to_per_pair_results


def get_bootstrapped_scores(values, n_resamples=100):
    for n in range(n_resamples):
        result = np.random.choice(values, size=len(values), replace=True).mean()
        yield result


def make_general_results_table(args):
    all_results = []
    for model in args.models:
        record = dict(
            Model=model.replace("_BASE", "").replace("_", "\_"),
        )
        path = f"runs/sentence-semantics/{model}/results"
        if args.alt_sentences:
            path += "_alt_sentences"
        path = path + ".csv"
        if os.path.isfile(path):
            results = pd.read_csv(path, index_col=False)
            results = transform_to_per_pair_results(results)

            acc = f"${results['result'].mean().round(2):.2f}$"
            record["Acc"] = acc

        path_cropped_images = f"runs/sentence-semantics/{model}/results_cropped_images"
        if args.alt_sentences:
            path_cropped_images += "_alt_sentences"
        path_cropped_images = path_cropped_images + ".csv"
        if os.path.isfile(path_cropped_images):
            results_cropped = pd.read_csv(path_cropped_images, index_col=False)
            results_cropped = transform_to_per_pair_results(results_cropped)

            acc_cropped = f"${results_cropped['result'].mean().round(2):.2f}$"
            record["Acc (cropped)"] = acc_cropped
        all_results.append(record)

    all_results = pd.DataFrame.from_records(all_results)
    all_results.to_latex("runs/sentence-semantics/results.tex", index=False, na_rep="", escape=False)


def make_noun_predicate_results_table(args):
    all_results = []
    for model in args.models:
        record = dict(
            Model=model.replace("_BASE", "").replace("_", "\_"),
        )
        path = f"runs/sentence-semantics/{model}/results"
        if args.alt_sentences:
            path += "_alt_sentences"
        path = path + ".csv"
        if os.path.isfile(path):
            results = pd.read_csv(path, index_col=False)
            results = transform_to_per_pair_results(results)

            results_noun = results[results.pos == "subject"]
            acc = f"${results_noun['result'].mean().round(2):.2f}$"
            record["Noun"] = acc
            results_predicate = results[results.pos == "object"]
            acc = f"${results_predicate['result'].mean().round(2):.2f}$"
            record["Predicate"] = acc

        path_cropped_images = f"runs/sentence-semantics/{model}/results_cropped_images"
        if args.alt_sentences:
            path_cropped_images += "_alt_sentences"
        path_cropped_images = path_cropped_images + ".csv"
        if os.path.isfile(path_cropped_images):
            results_cropped = pd.read_csv(path_cropped_images, index_col=False)
            results_cropped = transform_to_per_pair_results(results_cropped)
            results_noun = results_cropped[results_cropped.pos == "subject"]
            acc = f"${results_noun['result'].mean().round(2):.2f}$"
            record["Noun (cropped)"] = acc

            results_predicate = results_cropped[results_cropped.pos == "object"]
            acc = f"${results_predicate['result'].mean().round(2):.2f}$"
            record["Predicate (cropped)"] = acc
        all_results.append(record)

    all_results = pd.DataFrame.from_records(all_results)
    all_results.to_latex("runs/sentence-semantics/results_noun_predicate.tex", index=False, na_rep="", escape=False)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--models", type=str, nargs="+")
    argparser.add_argument("--alt-sentences", action="store_true", help="Evaluate using alternative sentences")

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    make_general_results_table(args)
    make_noun_predicate_results_table(args)
