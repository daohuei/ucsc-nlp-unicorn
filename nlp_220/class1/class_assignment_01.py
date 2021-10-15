import re

import nltk
from nltk import FreqDist
from nltk.tokenize import TweetTokenizer
from nltk.corpus import reuters
from nltk.corpus import twitter_samples

nltk.download("twitter_samples")
nltk.download("reuters")
nltk.download("averaged_perceptron_tagger")

# get all the tweets out from the corpus
tweets = twitter_samples.strings()

# get rid of the URLs and handles with regex substitution
cleaned_tweets = []
for tweet in tweets:
    # URLs
    result = re.sub(r"http\S+", "", tweet)
    # handles
    result = re.sub(r"@.*", "", result)
    cleaned_tweets.append(result)

# Tokenization with the Tweet Tokenizer
tweet_tokenizer = TweetTokenizer()
tokens = []
for cleaned_tweet in cleaned_tweets:
    # transfer to the lowercase before tokenization
    token = tweet_tokenizer.tokenize(cleaned_tweet.lower())
    # if token exist, put into the list
    if token:
        tokens.append(token)

# POS tagging
tagged_tokens = [nltk.pos_tag(token) for token in tokens]

# Count the Nouns and Adjectives
adj_count = 0
noun_count = 0
noun_words = []
for token in tagged_tokens:
    for word in token:
        if word:
            if "NN" in word[1]:
                noun_count += 1
                noun_words.append(word)
            if "JJ" in word[1]:
                adj_count += 1

print("Noun Count:", noun_count)
print("Adjective Count:", adj_count)

# Get the top 10 Noun with FreqDist
work_dist = nltk.FreqDist(noun_words)
print("Top 10 Nouns from Twitter Samples", work_dist.most_common(10))

# Extract all the category tags and put into a list
all_categories = []
for file_id in reuters.fileids():
    for category in reuters.categories(file_id):
        all_categories.append(category)

# calculate the frequency distribution
category_dist = FreqDist(all_categories)
print("Top 5 categories from Reuters corpus", category_dist.most_common(5))
