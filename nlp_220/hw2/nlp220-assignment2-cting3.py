"""
@File:      nlp220-assignment2-cting3.py
@Time:      2021/10/15
@Author:    Chih-Kai(Ken) Ting
@Contact:   cting3@ucsc.edu | tingken0214@gmail.com
@Desc:      Homework assignment 02 for NLP220
"""
import re
import json
from xml.dom.minidom import parse
import xml.dom.minidom
import xml.etree.ElementTree as ET

import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Part 1

# for removing non-alphabetic words and stopwords
def is_alphabetic(w):
    alphabetic_regex = "^[a-zA-Z]+$"
    return bool(re.match(alphabetic_regex, w))


def clean_data(words, lower=True):
    stop_tokens = set(stopwords.words("english"))
    return [
        (w.lower() if lower else w)
        for w in words
        if w not in stop_tokens and is_alphabetic(w)
    ]


# Load the ISEAR corpus
isear_df = pd.read_csv("ISEAR.csv", names=["emotion", "sentence", ""])
isear_df["tokens"] = [
    clean_data(word_tokenize(sent)) for sent in isear_df["sentence"]
]


# calculate maximum, minimum, average length of sentences for each emotion type (after removing stopwords and non-alphabetic words)
emotion_types = list(isear_df["emotion"].unique())
# a figure with a len(emotion_types)x1 grid of Axes
fig, axs = plt.subplots(
    len(emotion_types), 1, figsize=(25, 25), tight_layout=True
)
len_df = pd.DataFrame(
    columns=["Emotion Name", "Max-length", "Min-length", "Avg-length"]
)
for i, emotion in enumerate(emotion_types):
    tokens_df = isear_df[isear_df["emotion"] == emotion]["tokens"]
    lens = [len(tokens) for tokens in tokens_df]
    max_len = max(lens)
    min_len = min(lens)
    avg_len = round(sum(lens) / len(lens), 2)
    print("Emotion ", emotion)
    print("Max", max_len)
    print("Min", min_len)
    print("Avg", avg_len)
    len_df = len_df.append(
        {
            "Emotion Name": emotion,
            "Max-length": max_len,
            "Min-length": min_len,
            "Avg-length": avg_len,
        },
        ignore_index=True,
    )

    all_tokens = [token for tokens in tokens_df for token in tokens]
    # Frequency of Distribution
    token_freq = FreqDist(all_tokens)
    freq_df = pd.DataFrame(
        token_freq.items(), columns=["word", "frequency"]
    ).sort_values(by=["frequency"], ascending=False)
    # Vocabulary Size
    print("Vocab Size:", token_freq.B())
    print("Frequency Distr.\n", freq_df[:50])
    print("==============")

    # Plot the cumulative frequency distribution of the most 50 frequent tokens
    axs[i].plot(freq_df["word"][:50], freq_df["frequency"][:50], "b")

    # set the locator for avoiding text overflow
    axs[i].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=50))
    axs[i].set_xlabel("Word Tokens")
    axs[i].set_ylabel("Frequency")
    axs[i].set_title(f"Frequency Distribution of {emotion} Tokens")

fig.savefig("emotion_freq_dist.png")

# Output to a CSV file
len_df.to_csv("emotion-sentence-length.csv", sep="\t", line_terminator="  ")

# Part 2
# Import Sem-eval dataset
semeval_df = pd.read_csv("semeval-2017-train.csv", delimiter="\t")


def is_hashtag(w):
    # regex for hashtag: contains '#'
    hashtag_regex = "#"
    return bool(re.match(hashtag_regex, w))


def is_url(w):
    # regex for URL: contains "http://" or "https://"
    url_regex = "http[s]?://"
    return bool(re.match(url_regex, w))


# Print the sentences which contains URLs and hashtags
print(
    semeval_df[
        semeval_df["text"].apply(lambda text: is_hashtag(text) or is_url(text))
    ]
)

# Print all neutral sentences (annotated with 0)
print(semeval_df[semeval_df["label"] == 0])

# Find the distribution of positive, negative and neutral sentences
sentiments = semeval_df["label"].unique()
sentiment_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
for sentiment in sentiments:
    tokens_df = [
        clean_data(word_tokenize(text))
        for text in semeval_df[semeval_df["label"] == sentiment]["text"]
    ]
    all_tokens = [token for tokens in tokens_df for token in tokens]
    print("Sentiment:", sentiment_map[sentiment])
    # Frequency of Distribution
    token_freq = FreqDist(all_tokens)
    # Vocabulary Size
    print("Vocab Size:", token_freq.B())
    freq_df = pd.DataFrame(
        token_freq.items(), columns=["word", "frequency"]
    ).sort_values(by=["frequency"], ascending=False)
    print("Frequency Distr.\n", freq_df[:50])
    print("==============")

# Part 3
# Load the XML file ‘movies-new.xml’ using Python DOM parsing
dom_tree = parse("movies-new.xml")

# Find all the movie title, year, rating, and description and print the output
collection = dom_tree.documentElement
movies = collection.getElementsByTagName("movie")
movie_list = []
for movie in movies:
    title = movie.getAttribute("title")
    year = movie.getElementsByTagName("year")[0].firstChild.nodeValue
    rating = movie.getElementsByTagName("rating")[0].firstChild.nodeValue
    description = movie.getElementsByTagName("description")[
        0
    ].firstChild.nodeValue
    print("Title:", title)
    print("Year:", year)
    print("Rating:", rating)
    print("Description:", description)
    print("========================================")
    movie_list.append(
        {
            "title": title,
            "year": year,
            "rating": rating,
            "description": description,
        }
    )

# Output them in JSON
with open("./movies.json", "w") as movie_file:
    json.dump(movie_list, movie_file, indent=4)

# Part 4
# Load the XML file ‘movies-new.xml’ using Element-Tree
tree = ET.parse("movies-new.xml")
root = tree.getroot()
# Find the movie title, and year of all Action movies and print them
for movie in root.findall("./genre[@category='Action']/decade/movie"):
    print("Title:", movie.get("title"))
    print("Year:", movie.find("./year").text)
    print("==================================")
