import re
import json
from dateutil import parser

import tweepy
import pandas as pd
import numpy as np
from thefuzz import fuzz
import matplotlib
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from sklearn.model_selection import StratifiedShuffleSplit
import pymongo
from pymongo import MongoClient


# Load the dataset(https://www.kaggle.com/crowdflower/twitter-airline-sentiment?select=Tweets.csv) using Pandas
tweet_df = pd.read_csv("Tweets.csv")
# remove columns like tweet_coord, tweet_created, user_timezone(but this actually needed), etc
tweet_df = tweet_df[
    [
        "tweet_id",
        "airline_sentiment",
        "negativereason",
        "airline",
        "text",
        "tweet_created",
        "tweet_location",
        "user_timezone",
    ]
]

plt.rcParams.update({"font.size": 12})

# Finding each airline
airline_types = tweet_df["airline"].unique()
for airline_type in airline_types:
    df_of_type = tweet_df[tweet_df["airline"] == airline_type]

    airline_type_len = len(df_of_type)
    sentiment_type_len = len(df_of_type["airline_sentiment"].unique())
    reason_type_len = len(df_of_type["negativereason"].unique())

    # Not sure we should include the NaN, but value_counts function in Pandas will drop it
    top_sentiment = df_of_type["airline_sentiment"].value_counts().index[0]
    top_reason = df_of_type["negativereason"].value_counts().index[0]

    top_sentiment_freq = df_of_type["airline_sentiment"].value_counts()[0]
    top_reason_freq = df_of_type["negativereason"].value_counts()[0]

    shortest_tweet_len = df_of_type["text"].map(lambda tweet: len(tweet)).min()
    longest_tweet_len = df_of_type["text"].map(lambda tweet: len(tweet)).max()

    print(f"Airline: {airline_type}")
    print(f"Total Count: {airline_type_len}")
    print(f"Number of unique Airline Sentiment: {sentiment_type_len}")
    print(f"Most Frequent Airline Sentiment: {top_sentiment}")
    print(
        f"Frequency of the most Frequent Airline Sentiment: {top_sentiment_freq}"
    )
    print(f"Number of unique Negative Reason: {reason_type_len}")
    print(f"Most Frequent Negative Reason: {top_reason}")
    print(f"Frequency of the most Frequent Negative Reason: {top_reason_freq}")
    print(f"Length of shortest tweet: {shortest_tweet_len}")
    print(f"Length of longest tweet: {longest_tweet_len}")

    # plot the text length frequency distribution
    tweet_lens = df_of_type["text"].map(lambda tweet: len(tweet))
    x_bins = [i * 5 + 1 for i in range(int(tweet_lens.max() / 5))]
    # the histogram of the data
    plt.hist(tweet_lens, x_bins)
    plt.xlabel("Tweet Length")
    plt.ylabel("Frequency")
    plt.title(f"{airline_type} Tweet Length distribution")
    plt.xlim(0, tweet_lens.max())
    plt.show()
    plt.savefig(f"{airline_type}_tweet_length_distribution.jpg")

    print("", end="\n==============\n")


# Plot tweet sentiment distribution per airline in a single grid-like plot. plotting the frequency
fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=True)
sentiment_types = tweet_df["airline_sentiment"].unique()
for idx, ax in enumerate(axs.flatten()):
    airline_type = airline_types[idx]
    df_of_type = tweet_df[tweet_df["airline"] == airline_type]
    sentiment = df_of_type["airline_sentiment"]
    sentiment_freq = sentiment.value_counts()
    # reorder the setiments to make sure every graph have the same order
    re_order_indexes = [
        list(sentiment_freq.index).index(s) for s in sentiment_types
    ]
    re_order_freqs = [sentiment_freq[idx] for idx in re_order_indexes]
    ax.bar(sentiment_types, re_order_freqs, color=["g", "b", "r"])
    ax.set_title(airline_type)
fig.show()
fig.savefig("tweet_sentiment_distribution.jpg")


# Build your own version of NLTK’s tokenizer
class MyOwnTokenizer:
    def __init__(self):
        # extract several pattern for words and 's/'ll/'ve/'re and n't for negative terms
        # extract all special symbol in the sentence with [^\w\s] which means extract non-word that stick with non-space
        # also not doing punctuation
        self.re_pattern = r"(n't|'[\w]*|[^\w\s])"

    def tokenize(self, sentence):
        # add space at the front and behind for the special cases then do the split on space
        return re.sub(self.re_pattern, r" \1 ", sentence).split()


# Print the differences for 5 examples where your tokenizer behaves differently from the NLTK word tokenizer
my_own_tokenizer = MyOwnTokenizer()
same_samples = [
    "I don't like to eat Cici's food (it is true)",
    "i , am cool,.",
    "it's",
    "don't",
    "isn't",
    "#hashtag",
    "This is the period.",
]

print("Same Behavior")
for sample in same_samples:
    print("Text:             ", sample)
    print("My Own Tokenizer: ", my_own_tokenizer.tokenize(sample))
    print("NLTK Tokenizer:   ", word_tokenize(sample))
    print("=======================\n")
diff_samples = [
    '"to be or not to be"',
    "U.S.A",
    "Mr. Ken and Mrs. Hana",
    "feel so sad....",
    "twenty-two",
    "==================",
]
print("Different Behavior")
for sample in diff_samples:
    print("Text:             ", sample)
    print("My Own Tokenizer: ", my_own_tokenizer.tokenize(sample))
    print("NLTK Tokenizer:   ", word_tokenize(sample))
    print("=======================\n")

# The discussion is in the "tokenizer_discussion.md" file


# Find the number of missing values for tweet_location and user_timezone field
miss_val_loc_len = tweet_df["tweet_location"].isnull().sum()
miss_val_time_len = tweet_df["user_timezone"].isnull().sum()

print('Number of Missing Values of "tweet_location": ', miss_val_loc_len)
print('Number of Missing Values of "user_timezone": ', miss_val_time_len)

# Drop the missing value of tweet_location and user_timezone
dropped_tweet_df = tweet_df
dropped_tweet_df = dropped_tweet_df.drop(
    dropped_tweet_df[dropped_tweet_df["tweet_location"].isnull()].index
)
dropped_tweet_df = dropped_tweet_df.drop(
    dropped_tweet_df[dropped_tweet_df["user_timezone"].isnull()].index
)

# Now look at the tweet_created field.
# When you parse the file using Python, do you see this as parsed as date or a string?
print(
    f"The tweet_created field is actually parsed as: {type(list(dropped_tweet_df['tweet_created'])[0])}"
)
# Write the code to properly parse this as date.


def convert_datetime_str(datetime_str):
    return parser.parse(datetime_str)


dropped_tweet_df["tweet_created"] = dropped_tweet_df["tweet_created"].apply(
    lambda datetime_str: convert_datetime_str(datetime_str)
)

# Find the total number of tweets which are from Philadelphia
# (Think about misspelling, think about spaces between characters).
# Find all different spellings of Philadelphia. //fuzzywuzzy


def match_score(word_1, word_2):
    score = fuzz.ratio(word_1, word_2)
    return score


dropped_tweet_df["phi_score"] = dropped_tweet_df["tweet_location"].apply(
    lambda loc: match_score(
        str(loc).replace(" ", "").lower(), "Philadelphia".lower()
    )
)
tweet_from_phi_df = dropped_tweet_df[dropped_tweet_df["phi_score"] > 60]
unique_phi_spelling = tweet_from_phi_df["tweet_location"].unique()

print(
    f"Total number of tweets which are from Philadelphia: {len(tweet_from_phi_df)}"
)
print(f"Different spellings of Philadelphia: {list(unique_phi_spelling)}")


# Divide the datasets into train, dev and test sets such that 70% samples are in training, 20% in test and 10% in dev set
total_count = len(tweet_df)
train_size = int(total_count * 0.7)
val_size = int(total_count * 0.2)
test_size = total_count - train_size - val_size

dataset_dict = {}

# Simple random sampling
train_simple_df = tweet_df.sample(n=train_size)
test_simple_df = tweet_df.drop(train_simple_df.index)
val_simple_df = test_simple_df.sample(n=val_size)
test_simple_df = test_simple_df.drop(val_simple_df.index)

dataset_dict["simple"] = {
    "train": train_simple_df,
    "dev": val_simple_df,
    "test": test_simple_df,
}


# Stratified random sampling
train_test_split = StratifiedShuffleSplit(
    n_splits=1, test_size=val_size + test_size, train_size=train_size
)
train_strat_idx, test_strat_idx = next(
    train_test_split.split(tweet_df, tweet_df["airline_sentiment"])
)
train_strat_df = tweet_df.iloc[list(train_strat_idx)]
test_strat_df = tweet_df.iloc[list(test_strat_idx)]

val_test_split = StratifiedShuffleSplit(
    n_splits=1, test_size=test_size, train_size=val_size
)
val_strat_idx, test_strat_idx = next(
    val_test_split.split(test_strat_df, test_strat_df["airline_sentiment"])
)
val_strat_df = test_strat_df.iloc[list(val_strat_idx)]
test_strat_df = test_strat_df.iloc[list(test_strat_idx)]

dataset_dict["stratified"] = {
    "train": train_strat_df,
    "dev": val_strat_df,
    "test": test_strat_df,
}

# Compute the number of examples and distributions of labels for each set.
for sample_method in ["simple", "stratified"]:
    print(f"{sample_method} sampling: ")
    for data_set_key in ["train", "dev", "test"]:
        data_df = dataset_dict[sample_method][data_set_key]
        num_examples = len(data_df)
        label_dist = data_df["airline_sentiment"].value_counts()

        # Save the data partitions in {simple, stratified}-{train, dev, test}.csv
        data_df.to_csv(f"{sample_method}-{data_set_key}.csv", index=False)

        print(f"{data_set_key} set")
        print(f"Number of examples: {num_examples}")
        print(f"Label Distribution:\n{label_dist}")
        print("---------------------")

    print("=================================\n")


print(
    "Please Get all credentials from https://developer.twitter.com/en/portal"
)
consumer_key = input("Consumer Key: ")
consumer_secret = input("Consumer Secret: ")
access_token = input("Access Token: ")
access_token_secret = input("Access Token Secret: ")
bearer_token = input("Bearer Token: ")

api = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=consumer_key,
    consumer_secret=consumer_secret,
    access_token=access_token,
    access_token_secret=access_token_secret,
)

# Use Twitter’s public API to crawl 500 tweets containing the keyword “covid-19” and “vaccination rate”.
user_map = {}

tweet_ids = []
tweet_texts = []
author_ids = []
author_names = []
author_usernames = []

for i in range(5):
    public_tweets = api.search_recent_tweets(
        "covid-19 vaccination rate", expansions=["author_id"], max_results=100
    )

    for tweet_author in public_tweets.includes["users"]:
        user_map[tweet_author.id] = {
            "name": tweet_author.name,
            "username": tweet_author.username,
        }

    for tweet in public_tweets.data:
        tweet_ids.append(tweet.id)
        tweet_texts.append(tweet.text)
        author_ids.append(tweet.author_id)
        author_names.append(user_map[tweet.author_id]["name"])
        author_usernames.append(user_map[tweet.author_id]["username"])


fetched_tweets_df = pd.DataFrame(
    {
        "tweet_ids": tweet_ids,
        "tweet_texts": tweet_texts,
        "author_ids": author_ids,
        "author_names": author_names,
        "author_usernames": author_usernames,
    }
)

# Output the tweets in a CSV file
fetched_tweets_df.to_csv("api_fetched_tweets.csv", index=False)

# Store the tweets to a database of your choice: MongoDB
# Install MongoDB: https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/#install-mongodb-community-edition
# Install PyMongo
# This code has been set up in my own machine
# To make this code work, you have to set up your own local MongoDB server on 27017 port
client = MongoClient("localhost", 27017)
db = client["nlp_220_hw4_db"]
tweet_collection = db["tweets"]

tweet_datas = json.loads(fetched_tweets_df.to_json(orient="records"))
# Insert into the tweet collection/table
result = tweet_collection.insert_many(tweet_datas)


# Use your own tokenizer to tokenize the tweet text into n-grams (1-gram, 2-gram and 3-gram).
def get_n_grams(tokens, n):
    if n <= 1:
        return tokens
    if n >= len(tokens):
        return [" ".join(tokens)]
    return [
        " ".join([tokens[i - j] for j in reversed(range(n))])
        for i in range(n - 1, len(tokens))
    ]


fetched_tweets_df["1-gram"] = fetched_tweets_df["tweet_texts"].apply(
    lambda text: my_own_tokenizer.tokenize(text)
)
fetched_tweets_df["2-gram"] = fetched_tweets_df["1-gram"].apply(
    lambda tokens: get_n_grams(tokens, n=2)
)
fetched_tweets_df["3-gram"] = fetched_tweets_df["1-gram"].apply(
    lambda tokens: get_n_grams(tokens, n=3)
)

flatten_tokens = lambda corpus: [
    token for tokens in corpus for token in tokens
]
unigram_freq = FreqDist(flatten_tokens(fetched_tweets_df["1-gram"])).items()
bigram_freq = FreqDist(flatten_tokens(fetched_tweets_df["2-gram"])).items()
trigram_freq = FreqDist(flatten_tokens(fetched_tweets_df["3-gram"])).items()

construct_df = lambda freq: pd.DataFrame(freq, columns=["tokens", "frequency"])
unigram_freq_df = construct_df(unigram_freq).sort_values(
    "frequency", ascending=False
)
bigram_freq_df = construct_df(bigram_freq).sort_values(
    "frequency", ascending=False
)
trigram_freq_df = construct_df(trigram_freq).sort_values(
    "frequency", ascending=False
)

# Plot the frequency distribution of the n-grams.
plt.rcParams.update({"font.size": 20})

fig, axs = plt.subplots(3, 1, figsize=(30, 30), sharey=True, tight_layout=True)
ngram_freqs = [unigram_freq_df, bigram_freq_df, trigram_freq_df]
for idx, ax in enumerate(axs.flatten()):
    freq_df = ngram_freqs[idx]
    ax.bar(freq_df["tokens"][:50], freq_df["frequency"][:50])
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=20))
    ax.set_title(f"{idx+1}-gram Frequency Distribution")

fig.savefig("n_gram_freq_dist.jpg")
