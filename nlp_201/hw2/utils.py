"""contains utilities of text preprocessing"""
from random import sample
from collections import Counter

from tokens import START_TOKEN, STOP_TOKEN, UNK_TOKEN


# load the sentences without tokenization
def load_sentences(filename):
    sentences = []
    with open(filename, "r") as f:
        sentences = [sent for sent in f]
    return sentences


# tokenize the raw text by simply splitting with the space and add the start and stop tokens
def tokenize(raw_corpus):
    return [[START_TOKEN] + sent.split() + [STOP_TOKEN] for sent in raw_corpus]


# get the vocab of freq of give corpus with Counter
def get_vocab_freq(corpus):
    counter = Counter([token for sent in corpus for token in sent])
    return dict(counter)


# replace the low freq words with unknown tokens
def get_corpus_with_unk(corpus, vocab_freq, freq_constraint=3):
    return [
        [
            (
                token
                if not is_unkown(vocab_freq[token], token, freq_constraint)
                else UNK_TOKEN
            )
            for token in sent
        ]
        for sent in corpus
    ]


# to check if the token need to be replaced as an <UNK>
def is_unkown(freq, token, freq_constraint):
    return (
        token is not START_TOKEN
        and token is not STOP_TOKEN
        and freq < freq_constraint
    )


def get_corpus_by_portion(corpus, portion):
    # sample with given portion
    assert portion <= 1 and portion > 0
    if portion == 1:
        return corpus
    return sample(corpus, int(len(corpus) * portion))
