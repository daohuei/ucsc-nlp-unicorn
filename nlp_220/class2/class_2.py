import csv
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

phrase_df = pd.read_csv(
    "dictionary.txt", delimiter="|", names=["phrase", "phrase ids"]
)
label_df = pd.read_csv("sentiment_labels.txt", delimiter="|")
result_df = phrase_df.join(
    label_df, on="phrase ids", lsuffix="_caller", rsuffix=""
)
print(
    result_df[result_df["sentiment values"] <= 0.2][
        result_df["sentiment values"] >= 0.1
    ]
)

interval_df = result_df[result_df["sentiment values"] <= 0.5][
    result_df["sentiment values"] >= 0.1
]
print(
    interval_df[
        interval_df["phrase"].apply(lambda x: len(word_tokenize(x)) > 2)
    ]
)

print(
    result_df[
        result_df["phrase"].apply(
            lambda x: "love" in x or "like" in x or "hate" in x
        )
    ]
)


def like_is_verb(sent):
    if "like" not in sent.lower():
        return False
    words = word_tokenize(sent)
    tags = nltk.pos_tag(words)
    for tag in tags:
        if tag[0].lower() == "like" and "VB" in tag[1]:
            return True
    return False


print(result_df[result_df["phrase"].apply(lambda x: like_is_verb(x))])
