"""
@File:      nlp220-assignment3-cting3.py
@Time:      2021/11/12
@Author:    Chih-Kai(Ken) Ting
@Contact:   cting3@ucsc.edu | tingken0214@gmail.com
@Desc:      Homework assignment 03 for NLP220
"""
import re
from random import choices

import numpy as np
import pandas as pd
import nltk
from nltk.collocations import *
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.text import TextCollection
from nltk.probability import FreqDist, MLEProbDist
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt

nltk.download("universal_tagset")
nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")
nltk.download("wordnet")

# Part 1
# Import Semeval dataset (https://www.kaggle.com/azzouza2018/semevaldatadets) using Python.

semeval_df = pd.read_csv("semeval-2017-train.csv", delimiter="\t")
output_df = semeval_df.copy()
output_df["processed_text"] = semeval_df["text"]

# Remove all URLs from the sentences
def remove_urls(sent):
    return re.sub(r"(http[s]?://[^\ ]*)", "", sent)


output_df["processed_text"] = output_df["processed_text"].apply(remove_urls)

# Remove sentences which have less than 4 tokens
def is_less_than_four(sent):
    return len(sent.split()) < 4


output_df = output_df[
    output_df["processed_text"].apply(lambda text: not is_less_than_four(text))
]

# Remove the tokens which starts with @
def remove_tokens_start_with_at(sent):
    return re.sub(r"(?=\s?)(@[^\ ]*)", "", sent)


output_df["processed_text"] = output_df["processed_text"].apply(
    remove_tokens_start_with_at
)

# Remove all hashtags
def remove_hashtags(sent):
    return re.sub(r"(?=\s?)(#[^\ ]*)", "", sent)


output_df["processed_text"] = output_df["processed_text"].apply(
    remove_hashtags
)

# Remove occurrences of characters that appear more than 3 times consecutively, keep max occurrence as 3. For example: Heeeeelooooo => Heeelooo, gooooo => gooo
def remove_more_than_three_char_from_token(token):
    new_token = ""
    count = 1
    for i, c in enumerate(token):
        if c == token[i - 1]:
            count += 1
        else:
            count = 1
        if count > 3:
            continue
        new_token += c
    return new_token


def remove_more_than_three_char_from_sent(sent):
    new_tokens = []
    for token in sent.split():
        new_tokens.append(remove_more_than_three_char_from_token(token))

    return " ".join(new_tokens)


output_df["processed_text"] = output_df["processed_text"].apply(
    remove_more_than_three_char_from_sent
)


# Identify the slang words using a slang dictionary (e.g.: https://github.com/rishabhverma17/sms_slang_translator/blob/master/slang.txt) and output as follows in a CSV file: original tweet, slang word -> replacement word, ......
slang_dict = {}
with open("slang.txt") as slang_file:
    for row in slang_file.readlines():
        cols = row.split("=")
        if len(cols) == 2:
            slang_dict[cols[0]] = cols[1].strip()


def convert_slang_in_sent(sent, slang_dict):
    pairs = []
    for token in sent.split():
        if token in slang_dict.keys():
            pairs.append(f"{token} -> {slang_dict[token]}")
    if pairs:
        return " ".join(pairs)
    else:
        return "none"


for sent in output_df["processed_text"]:
    convert_slang_in_sent(sent, slang_dict)

slang_output_df = output_df
slang_output_df["slang word -> replacement word"] = slang_output_df[
    "processed_text"
].apply(lambda text: convert_slang_in_sent(text, slang_dict))
slang_output_df = slang_output_df.rename(columns={"text": "original tweet"})[
    ["original tweet", "slang word -> replacement word"]
]
slang_output_df.to_csv("slang_output.csv", index=False)

# Part 2

output_df["tokens"] = [
    word_tokenize(text) for text in output_df["processed_text"]
]
output_df["pos"] = nltk.pos_tag_sents(output_df["tokens"], tagset="universal")

# Now for each text, apply data augmentation to generate variations (max 5 variations per text). Use the following approach to generate variations:

# Replace nouns and verbs with their synonyms. Use wordnet for getting synonyms (https://www.nltk.org/howto/wordnet.html).


def is_noun(tag):
    return tag == "NOUN"


def is_verb(tag):
    return tag == "VERB"


def get_syn(pos_token):
    token = pos_token[0]
    tag = pos_token[1]
    if is_noun(tag):
        return wn.synsets(token, pos=wn.NOUN)
    if is_verb(tag):
        return wn.synsets(token, pos=wn.VERB)
    return []


def random_pick_syn(syns, k=1):
    picked_syn = []
    for syn_set in syns:
        if len(syn_set):
            picked_syn.append(choices(list(syn_set), k=k))
        else:
            picked_syn.append([])
    return picked_syn


def is_syn_empty(syns):
    total = 0
    for syn in syns:
        total += len(syn)
    return total == 0


def augment_data(pos_tag, k=5):
    syns = []
    for pos_token in pos_tag:
        syns.append(get_syn(pos_token))

    if is_syn_empty(syns):
        return [""] * k

    syns = [
        set([lemma.name() for synset in syn for lemma in synset.lemmas()])
        for syn in syns
    ]
    picked_syns = random_pick_syn(syns, k=k)
    augmented_data = []
    for i in range(k):
        new_tokens = []
        for j, picked_syn in enumerate(picked_syns):
            if len(picked_syn):
                new_tokens.append(picked_syn[i])
            else:
                new_tokens.append(pos_tag[j][0])
        new_text = " ".join(new_tokens)
        augmented_data.append(new_text)
    return augmented_data


augment_texts = [augment_data(pos_tag) for pos_tag in output_df["pos"]]
augment_texts_T = np.array(augment_texts).transpose()

output_df["augmentation1"] = list(augment_texts_T[0])
output_df["augmentation2"] = list(augment_texts_T[1])
output_df["augmentation3"] = list(augment_texts_T[2])
output_df["augmentation4"] = list(augment_texts_T[3])
output_df["augmentation5"] = list(augment_texts_T[4])

# Fill the empty cell with "None"
output_df["augmentation1"][output_df["augmentation1"] == ""] = "None"
output_df["augmentation2"][output_df["augmentation2"] == ""] = "None"
output_df["augmentation3"][output_df["augmentation3"] == ""] = "None"
output_df["augmentation4"][output_df["augmentation4"] == ""] = "None"
output_df["augmentation5"][output_df["augmentation5"] == ""] = "None"

# Output the original text and augmented text in a csv file using the following format:

# original text,  augmentation1, augmentation2, augmentation3, augmentation4, augmentation5
augment_output_df = output_df[
    [
        "text",
        "augmentation1",
        "augmentation2",
        "augmentation3",
        "augmentation4",
        "augmentation5",
    ]
].rename(columns={"text": "original text"})
augment_output_df.to_csv("augment_output.csv")

# Part 3
# size of your augmented dataset
print("size of your augmented dataset", len(output_df))

augmented_semeval_df = output_df[["label", "processed_text"]]
for col in [
    "augmentation1",
    "augmentation2",
    "augmentation3",
    "augmentation4",
    "augmentation5",
]:
    new_df = output_df[["label", col]].rename(columns={col: "processed_text"})
    augmented_semeval_df = augmented_semeval_df.append(
        new_df, ignore_index=True
    )

# ratio of original/augmented set
print(
    "ratio of original/augmented set:",
    len(output_df) / len(augmented_semeval_df),
)

# label distribution of the augmented set
label_freq_dist = FreqDist(list(augmented_semeval_df["label"]))
prob_dist = MLEProbDist(label_freq_dist)
print({label: prob_dist.prob(label) for label in prob_dist.samples()})

# plot the label probability distribution
fig, ax = plt.subplots()
x = prob_dist.samples()
y = [prob_dist.prob(sample) for sample in x]
bars = ax.bar([str(label) for label in x], y, align="center")

ax.set_xlabel("Labels")
ax.set_ylabel("Probability Distribution")
ax.set_title("Label distribution of the augmented set")
ax.set_ylim((0, 1))

fig.savefig("labels_prob_dist.png")

# - Generate n-gram (unigram and bigram, trigram) for the augmented dataset. Print them to console
def ngrams(tokens, n=2):
    for idx in range(len(tokens) - n + 1):
        yield tuple(tokens[idx : idx + n])


augmented_semeval_df["tokens"] = [
    word_tokenize(text) for text in augmented_semeval_df["processed_text"]
]
augmented_semeval_df["unigram"] = augmented_semeval_df["tokens"]
augmented_semeval_df["bigram"] = [
    [bigram for bigram in ngrams(tokens, n=2)]
    for tokens in augmented_semeval_df["tokens"]
]
augmented_semeval_df["trigram"] = [
    [trigram for trigram in ngrams(tokens, n=3)]
    for tokens in augmented_semeval_df["tokens"]
]

print(
    augmented_semeval_df[
        ["processed_text", "unigram", "bigram", "trigram"]
    ].to_string(max_rows=10, max_colwidth=40)
)

# Part 4
# Now compute rank/frequency profile of words of the original corpus and augmented corpus.
# To get the rank/frequency profile, you take the type from the frequency list and replace it with its rank, where the most frequent type is given rank 1, and so forth.
augmented_tokens = [
    token for tokens in augmented_semeval_df["tokens"] for token in tokens
]
original_tokens = [token for tokens in output_df["tokens"] for token in tokens]

augmented_freq_dist = FreqDist(augmented_tokens)
original_freq_dist = FreqDist(original_tokens)
augmented_freq_df = pd.DataFrame(
    augmented_freq_dist.items(), columns=["token", "frequency"]
).sort_values("frequency", ascending=False)
original_freq_df = pd.DataFrame(
    original_freq_dist.items(), columns=["token", "frequency"]
).sort_values("frequency", ascending=False)
augmented_freq_df["rank"] = list(range(1, len(augmented_freq_df) + 1))
original_freq_df["rank"] = list(range(1, len(original_freq_df) + 1))

# Print the rank/frequency profile of the words in csv files for both corpus as follows:
# rank1, freq 1
# rank 2, freq 2
# ......
augmented_freq_df["word"] = augmented_freq_df["token"]
original_freq_df["word"] = original_freq_df["token"]
print(
    "Augmented rank/frequency \n",
    augmented_freq_df[["word", "rank", "frequency"]].to_string(
        index=False, max_rows=10
    ),
)
print("\n")
print(
    "Original rank/frequency \n",
    original_freq_df[["word", "rank", "frequency"]].to_string(
        index=False, max_rows=10
    ),
)
augmented_freq_df[["word", "rank", "frequency"]].to_csv(
    "augmented_rank_frequency.csv", index=False
)
original_freq_df[["word", "rank", "frequency"]].to_csv(
    "original_rank_frequency.csv", index=False
)

# Output the percentage of corpus size made up by the top-10 words for both original and augmented corpus.
def is_made_up_by_words(tokens, words):
    for token in tokens:
        if token in words:
            return True
    return False


def count_made_up_by_top_10(df, freq_dist):
    count = 0
    top_10_words = list(map(lambda pair: pair[0], freq_dist.most_common(10)))
    for tokens in df["tokens"]:
        if is_made_up_by_words(tokens, top_10_words):
            count += 1
    return count


augmented_count = count_made_up_by_top_10(
    augmented_semeval_df, augmented_freq_dist
)
original_count = count_made_up_by_top_10(output_df, original_freq_dist)

print(
    "the percentage of corpus size made up by the top-10 words for original and augmented corpus"
)
print("original", original_count / len(output_df) * 100, "%")
print("augmented", augmented_count / len(augmented_semeval_df) * 100, "%")

# Part 5
# Find top-10 bi-grams and tri-grams from positive sentiment samples and top-10 bi-grams and tri-grams from negative sentiment samples using NLTK’s significant collocation approach.
# Investigate the difference of these n-grams when you try:

# PMI to select significant collocations (Review materials from lecture 6)
# Maximum likelihood to select significant collocations    (use likelihood ratio. Review materials from lecture 6)


bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

positive_tokens = [
    token
    for tokens in augmented_semeval_df[augmented_semeval_df["label"] == 1][
        "tokens"
    ]
    for token in tokens
]
negative_tokens = [
    token
    for tokens in augmented_semeval_df[augmented_semeval_df["label"] == -1][
        "tokens"
    ]
    for token in tokens
]

positive_bigram_finder = BigramCollocationFinder.from_words(positive_tokens)
negative_bigram_finder = BigramCollocationFinder.from_words(negative_tokens)
positive_trigram_finder = TrigramCollocationFinder.from_words(positive_tokens)
negative_trigram_finder = TrigramCollocationFinder.from_words(negative_tokens)

print(
    "Postive Top-10 Bigram with PMI",
    positive_bigram_finder.nbest(bigram_measures.pmi, 10),
)
print(
    "Negative Top-10 Bigram with PMI",
    negative_bigram_finder.nbest(bigram_measures.pmi, 10),
)
print(
    "Postive Top-10 Trigram with PMI",
    positive_trigram_finder.nbest(trigram_measures.pmi, 10),
)
print(
    "Negative Top-10 Trigram with PMI",
    negative_trigram_finder.nbest(trigram_measures.pmi, 10),
)

print(
    "Postive Top-10 Bigram with MLE",
    positive_bigram_finder.nbest(bigram_measures.likelihood_ratio, 10),
)
print(
    "Negative Top-10 Bigram with MLE",
    negative_bigram_finder.nbest(bigram_measures.likelihood_ratio, 10),
)
print(
    "Postive Top-10 Trigram with MLE",
    positive_trigram_finder.nbest(trigram_measures.likelihood_ratio, 10),
)
print(
    "Negative Top-10 Trigram with MLE",
    negative_trigram_finder.nbest(trigram_measures.likelihood_ratio, 10),
)

"""
Difference on two ngram finder measurements:
The top 10 ngram that generated with likelihood measure are more depends on the probability distribution of such ngram,
so top ngram are sequences and focus on high probability (frequency) ngrams.
With pointwise mutual information, it consider the cooccurance of words in the ngrams, which usually generate ngrams that are meaningless.
However, it will be more focus on words that may be truely relative to each others.
"""

# Part 6
#   Compute word vectors from this data-set using the following approaches:
#      1) Frequency based
#      2) One-hot
#      3) TF-IDF
#    Print the word vectors

# For preventing out-of-memory, we define vocab to be words larger than 100 frequency
freq_dist = FreqDist(
    [token for tokens in output_df["tokens"] for token in tokens]
)
freq_dist_df = pd.DataFrame(freq_dist.items(), columns=["token", "frequency"])
freq_dist_df = freq_dist_df[freq_dist_df["frequency"] > 100]
vocab = list(freq_dist_df["token"])

# Frequency based
count_vectorizer = CountVectorizer(vocabulary=vocab)
bow_vectors = count_vectorizer.fit_transform(
    output_df["processed_text"]
).toarray()


# One Hot
onehot_vectorizer = Binarizer()
one_hot_vectors = onehot_vectorizer.fit_transform(bow_vectors)

#  TF-IDF
tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab)
tfidf_vectors = tfidf_vectorizer.fit_transform(
    output_df["processed_text"]
).toarray()

print("Frequency-based", bow_vectors)
print("One-Hot", one_hot_vectors)
print("TF-IDF", tfidf_vectors)
