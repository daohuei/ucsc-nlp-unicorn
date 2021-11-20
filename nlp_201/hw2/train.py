"""function to train the ngram, defined training functions with different smoothing methods"""
from ngram import *


def train_regular(corpus, n_gram=1):
    # Basic
    unigram = Unigram()
    unigram.train_unigram(corpus)
    if n_gram == 1:
        return unigram

    bigram = Bigram()
    bigram.set_unigram(unigram)
    bigram.train_bigram(corpus)

    if n_gram == 2:
        return bigram

    trigram = Trigram()
    trigram.set_bigram(bigram)
    trigram.train_trigram(corpus)

    if n_gram == 3:
        return trigram

    return None


def train_additive(corpus, n_gram=1, alpha=1):

    assert alpha > 0

    unigram = Unigram()
    unigram.train_unigram(corpus, smoothing="additive", alpha=alpha)
    if n_gram == 1:
        return unigram

    bigram = Bigram()
    bigram.set_unigram(unigram)
    bigram.train_bigram(corpus, smoothing="additive", alpha=alpha)

    if n_gram == 2:
        return bigram

    trigram = Trigram()
    trigram.set_bigram(bigram)
    trigram.train_trigram(corpus, smoothing="additive", alpha=alpha)

    if n_gram == 3:
        return trigram

    return None


def train_interpolation(corpus, pretrained_bigram, l1=0.1, l2=0.3, l3=0.6):

    trigram_interpolation = Trigram()
    trigram_interpolation.set_bigram(pretrained_bigram)
    trigram_interpolation.train_trigram(
        corpus,
        smoothing="interpolation",
        l1=l1,
        l2=l2,
        l3=l3,
    )
    return trigram_interpolation


def get_pretrained_bigram(corpus):
    return train_regular(corpus, n_gram=2)
