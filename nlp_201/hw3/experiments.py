import os

import matplotlib.pyplot as plt

from taggers import HMMTagger, BaselineTagger
from utils import *
from starter_code import evaluate, load_treebank_splits

# Set path for datadir
datadir = os.path.join("data", "penn-treebank3-wsj", "wsj")
# load the dataset from directory
train, dev, test = load_treebank_splits(datadir)

# extract the gold from dev/test set
gold_dev = preprocess_corpus(dev, preprocess_labeled_text)
gold_test = preprocess_corpus(test, preprocess_labeled_text)

# the train set contained labels
train_corpus = train

# the dev/test set without labels
dev_corpus = preprocess_corpus(
    gold_dev,
    preprocess_extract_words,
)

test_corpus = preprocess_corpus(
    gold_test,
    preprocess_extract_words,
)

# remove special tokens for golds
gold_dev = preprocess_corpus(gold_dev, preprocess_remove_start_stop_tokens)
gold_test = preprocess_corpus(gold_test, preprocess_remove_start_stop_tokens)

# using small amount of samples for training and testing
def experiment_debug(alpha=1):
    train_samples = [
        "The/DT House/NNP joined/VBD  the/DT Senate/NNP in/IN making/VBG  federal/JJ reparations/NNS for/IN Japanese-Americans/NNPS"
    ]
    test_samples = [
        "alsjfla the askdmc and djdsaas in making",
        "vsacs the House and djdsaas in giuhun",
        "The House joined  the Senate in making  federal reparations for Japanese-Americans",
    ]
    tagger = HMMTagger()
    tagger.train(train_samples, alpha)
    hmm_y = tagger.predict(test_samples)

    b_tagger = BaselineTagger()
    b_tagger.train(train_corpus)
    b_y = b_tagger.predict(test_samples)

    hmm_y = preprocess_corpus(hmm_y, preprocess_remove_start_stop_tokens)
    b_y = preprocess_corpus(b_y, preprocess_remove_start_stop_tokens)
    print(evaluate(hmm_y, b_y))


def experiment_baseline():
    baseline_tagger = BaselineTagger()
    baseline_tagger.train(train_corpus)

    # dev
    prediction_dev = baseline_tagger.predict(dev_corpus)
    prediction_dev = preprocess_corpus(
        prediction_dev, preprocess_remove_start_stop_tokens
    )
    print(evaluate(gold_dev, prediction_dev))

    # test
    prediction_test = baseline_tagger.predict(test_corpus)
    prediction_test = preprocess_corpus(
        prediction_test, preprocess_remove_start_stop_tokens
    )
    print(evaluate(gold_test, prediction_test))


def experiment_hmm(alpha=1):
    hmm_tagger = HMMTagger()
    hmm_tagger.train(train_corpus, alpha)

    # dev
    prediction_dev = hmm_tagger.predict(dev_corpus)
    prediction_dev = preprocess_corpus(
        prediction_dev, preprocess_remove_start_stop_tokens
    )
    print(evaluate(gold_dev, prediction_dev))

    # test
    prediction_test = hmm_tagger.predict(test_corpus)
    prediction_test = preprocess_corpus(
        prediction_test, preprocess_remove_start_stop_tokens
    )
    print(evaluate(gold_test, prediction_test))


def experiment_hmm_alpha(alpha=1):
    hmm_tagger = HMMTagger()
    hmm_tagger.train(train_corpus, alpha)

    # dev
    prediction_dev = hmm_tagger.predict(dev_corpus[0:5])
    prediction_dev = preprocess_corpus(
        prediction_dev, preprocess_remove_start_stop_tokens
    )
    return evaluate(gold_dev[0:5], prediction_dev, output_dict=True)


def plot_hmm_param_performance():

    fig, axs = plt.subplots(3, 1, figsize=(15, 10), tight_layout=True)

    accuracy_ax, macro_ax, weighted_ax = (
        axs[0],
        axs[1],
        axs[2],
    )

    alphas = [1, 0.001]
    accuracy_list = []
    macro_precision_list = []
    macro_recall_list = []
    macro_f1_list = []
    weighted_precision_list = []
    weighted_recall_list = []
    weighted_f1_list = []

    accuracy_ax.set_xscale("log")
    macro_ax.set_xscale("log")
    weighted_ax.set_xscale("log")

    accuracy_ax.set_xlabel("alpha")
    macro_ax.set_xlabel("alpha")
    weighted_ax.set_xlabel("alpha")

    accuracy_ax.set_title("Accuracy")
    macro_ax.set_title("Macro Avg")
    weighted_ax.set_title("Weighted Avg")

    accuracy_ax.set_ylim(0, 1)
    macro_ax.set_ylim(0, 1)
    weighted_ax.set_ylim(0, 1)

    for alpha in alphas:
        report = experiment_hmm_alpha(alpha)
        accuracy_list.append(report["accuracy"])
        macro_precision_list.append(report["macro avg"]["precision"])
        macro_recall_list.append(report["macro avg"]["recall"])
        macro_f1_list.append(report["macro avg"]["f1-score"])
        weighted_precision_list.append(report["weighted avg"]["precision"])
        weighted_recall_list.append(report["weighted avg"]["recall"])
        weighted_f1_list.append(report["weighted avg"]["f1-score"])

    accuracy_ax.plot(alphas, accuracy_list, "g")
    macro_ax.plot(alphas, macro_precision_list, "b", label="precision")
    macro_ax.plot(alphas, macro_recall_list, "r", label="recall")
    macro_ax.plot(alphas, macro_f1_list, "g", label="f1-score")
    weighted_ax.plot(alphas, weighted_precision_list, "b", label="precision")
    weighted_ax.plot(alphas, weighted_recall_list, "r", label="recall")
    weighted_ax.plot(alphas, weighted_f1_list, "g", label="f1-score")

    macro_ax.legend()
    weighted_ax.legend()

    fig.savefig("alpha_report.png")


plot_hmm_param_performance()