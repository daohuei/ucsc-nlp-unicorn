"""contains utilities of text preprocessing"""
from tokens import *
from starter_code import get_token_tag_tuples


def preprocess_raw_text(sent):
    return [START_TOKEN] + sent.split() + [STOP_TOKEN]


def preprocess_labeled_text(sent):
    return (
        [(START_TOKEN, START_TOKEN)]
        + get_token_tag_tuples(sent)
        + [(STOP_TOKEN, STOP_TOKEN)]
    )


def preprocess_remove_start_stop_tokens(sent):
    return sent[1:-1]


def preprocess_extract_words(sent):
    return [word for word, _ in sent]


def preprocess_extract_tags(sent):
    return [tag for _, tag in sent]


def preprocess_corpus(corpus, preprocess):
    return [preprocess(sent) for sent in corpus]


def preprocess_flatten_corpus(corpus, preprocess):
    return [token for sent in corpus for token in preprocess(sent)]