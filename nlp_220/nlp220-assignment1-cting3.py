"""
@File:      nlp220-assignment1-cting3.py
@Time:      2021/10/9
@Author:    Chih-Kai(Ken) Ting
@Contact:   cting3@ucsc.edu | tingken0214@gmail.com
@Desc:      Homework assignment 01 for NLP220
"""
import re

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from nltk.corpus import reuters
from nltk.corpus import twitter_samples
from nltk.probability import FreqDist
import spacy


# download corpus
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("reuters")
nltk.download("gutenberg")
nltk.download("twitter_samples")

# Part 1


def is_alphabetic(w):
    # the regex that contains any alphabetic character
    alphabetic_regex = ".*[a-zA-Z]+.*"
    return bool(re.match(alphabetic_regex, w))


def clean_data(words, lower=True):
    stop_tokens = set(stopwords.words("english"))
    # output lowercase(default, but can be disabled), non-stopword, and is alphabetic tokens
    return [
        (w.lower() if lower else w)
        for w in words
        if w not in stop_tokens and is_alphabetic(w)
    ]


# pick a book
file_id = "austen-emma.txt"
# corpus-based tokenization
words = gutenberg.words(file_id)
# remove stopwords and keep words that contain alphabet
words = clean_data(words)
# Compute the vocabulary of the book with frequency distribution
vocab_freq_dist = FreqDist(words)
# load frequency as pandas dataframe
vocab_df = pd.DataFrame.from_dict(
    vocab_freq_dist, orient="index", columns=["frequency"]
)
# assign the token column with index value(which should be token as well)
vocab_df["token"] = vocab_df.index
# organize the order of columns
vocab_df = vocab_df[["token", "frequency"]]
# sort tokens by frequency in descending order
vocab_df = vocab_df.sort_values(by=["frequency"], ascending=False)
# output as a csv file
vocab_df.to_csv("./vocab_freq_gutenberg.csv", sep=",", index=False)

# determine the POS tags for tokens
tagged_words = nltk.pos_tag(words)
# find the frequency distribution of the POS tags
pos_tag_freq_dist = FreqDist([w[1] for w in tagged_words])
# load frequency as pandas dataframe
pos_tag_df = pd.DataFrame.from_dict(
    pos_tag_freq_dist, orient="index", columns=["frequency"]
)
# assign the pos_tag column with index value(which should be pos_tag as well)
pos_tag_df["pos_tag"] = pos_tag_df.index
# organize the order of columns
pos_tag_df = pos_tag_df[["pos_tag", "frequency"]]
# sort pos_tag by frequency in descending order
pos_tag_df = pos_tag_df.sort_values(by=["frequency"], ascending=False)

# a figure with a 2x1 grid of Axes
fig, axs = plt.subplots(2, 1, figsize=(10, 10), tight_layout=True)
vocab_ax, pos_ax = axs[0], axs[1]

# Plot the cumulative frequency distribution of the most 50 frequent tokens
vocab_ax.plot(vocab_df["token"][:50], vocab_df["frequency"][:50].cumsum(), "b")

# set the locator for avoiding text overflow
vocab_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=10))
vocab_ax.set_xlabel("Word Tokens")
vocab_ax.set_ylabel("Frequency")
vocab_ax.set_title("Cumulative Frequency Distribution of Tokens")

# Plot the simple frequency distribution of the most 50 frequent POS tags
pos_ax.plot(pos_tag_df["pos_tag"][:50], pos_tag_df["frequency"][:50], "r")
# set the locator for avoiding text overflow
pos_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=10))
pos_ax.set_xlabel("POS Tags")
pos_ax.set_ylabel("Frequency")
pos_ax.set_title("Simple Frequency Distribution of POS Tags")

# save the figure as a png image file
fig.savefig("freq_dist_for_part_1.png")

# init the spacy english parser
# disable tagger and parser pipeline for saving runtime
# we only need the NER pipeline
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])
# the results will be different if we lower all cases, so we need to use another tokens that not doing lower processing
ner_words = clean_data(gutenberg.words(file_id), lower=False)
# load the raw document with spacy
doc = nlp(" ".join(ner_words))
# generate frequency distribution of "PERSON" entity
person_freq_dist = FreqDist()
for ent in doc.ents:
    if ent.label_ == "PERSON":
        person_freq_dist[ent.text] += 1

# output the most 50 frequent names in the book
print("Top-50 frequent person name: ", person_freq_dist.most_common(50))

# Part 2

# load the Reuters words
reuters_words = reuters.words()
# clean the words
reuters_words = clean_data(reuters_words)
# Compute the frequency distribution of word tokens
reuters_freq_dist = FreqDist(reuters_words)
# load frequency as pandas dataframe
reuters_freq_df = pd.DataFrame.from_dict(
    reuters_freq_dist, orient="index", columns=["frequency"]
)
# assign the token column with index value(which should be token as well)
reuters_freq_df["token"] = reuters_freq_df.index
# organize the order of columns
reuters_freq_df = reuters_freq_df[["token", "frequency"]]
# sort tokens by frequency in descending order
reuters_freq_df = reuters_freq_df.sort_values(
    by=["frequency"], ascending=False
)

# a figure with a 2x1 grid of Axes
reuters_fig, reuters_axes = plt.subplots(
    2, 1, figsize=(10, 10), tight_layout=True
)
reuters_vocab_ax, reuters_topic_ax = reuters_axes[0], reuters_axes[1]

# Plot the simple frequency distribution of the most 50 frequent tokens
reuters_vocab_ax.plot(
    reuters_freq_df["token"][:50], reuters_freq_df["frequency"][:50], "g"
)
# set the locator for avoiding text overflow
reuters_vocab_ax.xaxis.set_major_locator(
    matplotlib.ticker.MaxNLocator(nbins=10)
)
reuters_vocab_ax.set_xlabel("Word Tokens")
reuters_vocab_ax.set_ylabel("Frequency")
reuters_vocab_ax.set_title("Frequency Distribution of Reuters Tokens")

# Output the 10 most frequent words in the reuters corpus
print("Top-10 words according to frequency\n", reuters_freq_df[:10])

# Calculate the frequency distribution
category_dist = FreqDist()
for file_id in reuters.fileids():
    category_dist.update(reuters.categories(file_id))

# Output the 10 most frequent topics in the reuters corpus
top_10_categories = category_dist.most_common(10)
category_df = pd.DataFrame.from_dict(
    {
        "category": [c[0] for c in top_10_categories],
        "frequency": [c[1] for c in top_10_categories],
    }
)
print("Top-10 topics from Reuters corpus\n", category_df)

# Plot the simple frequency distribution of the most 10 frequent topics
reuters_topic_ax.plot(
    category_df["category"][:10], category_df["frequency"][:10], "y"
)
# set the locator for avoiding text overflow
reuters_topic_ax.xaxis.set_major_locator(
    matplotlib.ticker.MaxNLocator(nbins=10)
)
reuters_topic_ax.set_xlabel("Topics")
reuters_topic_ax.set_ylabel("Frequency")
reuters_topic_ax.set_title("Frequency Distribution of Reuters Topics")

# Save the figure as a png image file
reuters_fig.savefig("freq_dist_for_part_2.png")

# Part 3


def is_hashtag(w):
    # regex for hashtag: start with '#'
    hashtag_regex = "^#.*"
    return bool(re.match(hashtag_regex, w))


def is_mention(w):
    # regex for mention: start with '@'
    mention_regex = "^@.*"
    return bool(re.match(mention_regex, w))


def is_url(w):
    # regex for URL: start with "http://" or "https://"
    url_regex = "^http[s]?://.*"
    return bool(re.match(url_regex, w))


# get the tokens from Tweet tokenizer
twitter_sents = twitter_samples.tokenized()
twitter_tokens = [token for sent in twitter_sents for token in sent]

# count the hashtags, mentions, and urls
hashtag_count = 0
mention_count = 0
url_count = 0
cleaned_token = []
for token in twitter_tokens:
    # remove the Hashtags and Mentions by ignoring it
    if is_hashtag(token):
        hashtag_count += 1
        continue
    elif is_mention(token):
        mention_count += 1
        continue
    elif is_url(token):
        url_count += 1
    # put the token in the list
    cleaned_token.append(token)

print("Hashtag count number: ", hashtag_count)
print("Metion count number: ", mention_count)
print("URLs count number: ", url_count)

print("output cleaned tokens: ", cleaned_token)
