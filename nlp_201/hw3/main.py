"""a script that can train the model with arguments, see usage section in readme.md"""
import os

from taggers import HMMTagger, BaselineTagger
from utils import *
from starter_code import evaluate, load_treebank_splits


if __name__ == "__main__":

    # Set path for datadir
    datadir = os.path.join("data", "penn-treebank3-wsj", "wsj")

    train, dev, test = load_treebank_splits(datadir)
    train_corpus = train
    gold = preprocess_corpus(dev, preprocess_labeled_text)
    dev_corpus = preprocess_corpus(
        gold,
        preprocess_extract_words,
    )

    baseline_tagger = BaselineTagger()
    baseline_tagger.train(train)
    prediction = baseline_tagger.predict(dev_corpus)

    # hmm_tagger = HMMTagger()
    # hmm_tagger.train(train)
    # prediction = hmm_tagger.predict(dev_corpus)

    gold = preprocess_corpus(gold, preprocess_remove_start_stop_tokens)
    prediction = preprocess_corpus(
        prediction, preprocess_remove_start_stop_tokens
    )

    print(evaluate(gold, prediction))