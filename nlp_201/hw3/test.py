import os

import nltk

from starter_code import load_treebank_splits, get_token_tag_tuples

# Set path for datadir
datadir = os.path.join("data", "penn-treebank3-wsj", "wsj")

train, dev, test = load_treebank_splits(datadir)


print(train[0])

train_sentences = [get_token_tag_tuples(sent) for sent in train]
