"""a script for running different experiments that mentioned in the problem set of homework 2"""
from train import *
from utils import *

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


def eval_lm(lm, name=""):
    print(
        f"{name}_train: {lm.perplexity(train_corpus)} {name}_dev: {lm.perplexity(val_corpus)} {name}_test: {lm.perplexity(test_corpus)}"
    )


# Problem 2-1:
def experiment_regular():
    trigram = train_regular(train_corpus_with_unk, n_gram=3)
    bigram = trigram.bigram
    unigram = trigram.bigram.unigram
    eval_lm(unigram, "unigram")
    eval_lm(bigram, "bigram")
    eval_lm(trigram, "trigram")


# Problem 3-1:
def experiment_add_one():
    trigram = train_additive(train_corpus_with_unk, n_gram=3, alpha=1)
    bigram = trigram.bigram
    unigram = trigram.bigram.unigram
    eval_lm(unigram, "unigram")
    eval_lm(bigram, "bigram")
    eval_lm(trigram, "trigram")


# Problem 3-2:
def experiment_add_k(k):
    trigram = train_additive(train_corpus_with_unk, n_gram=3, alpha=k)
    bigram = trigram.bigram
    unigram = trigram.bigram.unigram
    eval_lm(unigram, "unigram")
    eval_lm(bigram, "bigram")
    eval_lm(trigram, "trigram")


# Problem 4-1 & 4-2:
def experiment_interpolation(l1, l2, l3):
    pretrained_bigram = get_pretrained_bigram(train_corpus_with_unk)
    trigram_interpolation = train_interpolation(
        train_corpus_with_unk, pretrained_bigram, l1, l2, l3
    )
    eval_lm(trigram_interpolation, "trigram_interpolation")


# Problem 4-3:
def experiment_half_data(l1, l2, l3):

    # get only a portion of training corpus
    half_of_train_corpus = get_corpus_by_portion(train_corpus, 0.5)

    # get the vocab/freq map for corpus
    vocab_freq_portion = get_vocab_freq(half_of_train_corpus)

    # replace low freq (according to the defined restriction) words with unknown tokens
    half_of_train_corpus_with_unk = get_corpus_with_unk(
        half_of_train_corpus, vocab_freq_portion
    )

    pretrained_bigram = get_pretrained_bigram(half_of_train_corpus_with_unk)
    trigram_interpolation = train_interpolation(
        half_of_train_corpus_with_unk, pretrained_bigram, l1, l2, l3
    )
    eval_lm(trigram_interpolation, "trigram_interpolation_half_corpus")


# Problem 4-4:
def experiment_low_freq(l1, l2, l3, freq_restriction, name=""):
    # replace low freq (according to the defined restriction) words with unknown tokens
    train_corpus_low_freq = get_corpus_with_unk(
        train_corpus, vocab_freq_all, freq_restriction
    )

    pretrained_bigram = get_pretrained_bigram(train_corpus_low_freq)
    trigram_low_freq = train_interpolation(
        train_corpus_low_freq, pretrained_bigram, l1, l2, l3
    )
    eval_lm(trigram_low_freq, f"trigram_interpolation_{name}")


print("without smoothing")
experiment_regular()

print("Add-1")
experiment_add_one()
print("Add-10")
experiment_add_k(10)
print("Add-0.1")
experiment_add_k(0.1)

print("l1=0.3, l2=0.6, l3=0.1")
experiment_interpolation(l1=0.3, l2=0.6, l3=0.1)
print("l1=0.3, l2=0.1, l3=0.6")
experiment_interpolation(l1=0.3, l2=0.1, l3=0.6)
print("l1=0.6, l2=0.3, l3=0.1")
experiment_interpolation(l1=0.6, l2=0.3, l3=0.1)
print("l1=0.6, l2=0.1, l3=0.3")
experiment_interpolation(l1=0.6, l2=0.1, l3=0.3)
print("l1=0.1, l2=0.6, l3=0.3")
experiment_interpolation(l1=0.1, l2=0.6, l3=0.3)
print("l1=0.1, l2=0.3, l3=0.6")
experiment_interpolation(l1=0.1, l2=0.3, l3=0.6)

print("l1=0.1, l2=0.3, l3=0.6 with half data")
experiment_half_data(l1=0.3, l2=0.6, l3=0.1)

print("l1=0.1, l2=0.3, l3=0.6 less than 5 unknown")
experiment_low_freq(
    l1=0.3, l2=0.6, l3=0.1, freq_restriction=5, name="less_than_5"
)
print("l1=0.1, l2=0.3, l3=0.6 only once unknown")
experiment_low_freq(
    l1=0.3, l2=0.6, l3=0.1, freq_restriction=2, name="only_once"
)
