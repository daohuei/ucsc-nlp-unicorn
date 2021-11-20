"""a script that can train the model with arguments, see usage section in readme.md"""
import sys

from utils import *
from ngram import *
from train import *


def print_debugging_score(train_corpus):
    # Debugging score
    unigram = Unigram()
    unigram.train_unigram(train_corpus)
    print(unigram.perplexity(tokenize(["HDTV ."])))

    bigram = Bigram()
    bigram.set_unigram(unigram)
    bigram.train_bigram(train_corpus)
    print(bigram.perplexity(tokenize(["HDTV ."])))

    trigram = Trigram()
    trigram.set_bigram(bigram)
    trigram.train_trigram(train_corpus)
    print(trigram.perplexity(tokenize(["HDTV ."])))


if __name__ == "__main__":
    args = sys.argv
    mode = args[1]

    # load every sentences without any tokenization
    train_sents = load_sentences("1b_benchmark.train.tokens")
    val_sents = load_sentences("1b_benchmark.dev.tokens")
    test_sents = load_sentences("1b_benchmark.test.tokens")

    # tokenize the corpus
    train_corpus = tokenize(train_sents)
    val_corpus = tokenize(val_sents)
    test_corpus = tokenize(test_sents)

    # get the vocab/freq map for all tokens
    vocab_freq_all = get_vocab_freq(train_corpus)
    # replace low freq words with unknown tokens
    train_corpus_with_unk = get_corpus_with_unk(train_corpus, vocab_freq_all)

    if mode == "debug":
        print_debugging_score(train_corpus_with_unk)

    n_gram = int(args[2])
    train_portion = float(args[3])
    freq_restriction = int(args[4])

    # get only a portion of training corpus
    portion_of_train_corpus = get_corpus_by_portion(
        train_corpus, train_portion
    )

    # get the vocab/freq map for corpus
    vocab_freq_portion = get_vocab_freq(portion_of_train_corpus)
    # replace low freq (according to the defined restriction) words with unknown tokens
    train_corpus_with_unk = get_corpus_with_unk(
        portion_of_train_corpus, vocab_freq_portion, freq_restriction
    )

    # language model
    lm = None
    if mode == "regular":
        lm = train_regular(train_corpus_with_unk, n_gram)

    if mode == "additive":
        alpha = float(args[5])
        lm = train_additive(train_corpus_with_unk, n_gram, alpha)

    if mode == "interpolation":
        l1 = float(args[5])
        l2 = float(args[6])
        l3 = float(args[7])
        lm = train_interpolation(
            train_corpus_with_unk,
            get_pretrained_bigram(train_corpus_with_unk),
            l1,
            l2,
            l3,
        )
    print("train perplexity:", lm.perplexity(train_corpus))
    print("dev perplexity:", lm.perplexity(val_corpus))
    print("test perplexity:", lm.perplexity(test_corpus))
