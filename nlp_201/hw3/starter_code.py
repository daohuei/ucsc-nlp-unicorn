import os
import nltk
from sklearn import metrics


def evaluate(test_sentences, tagged_test_sentences, output_dict=False):
    gold = [str(tag) for sentence in test_sentences for token, tag in sentence]
    pred = [
        str(tag)
        for sentence in tagged_test_sentences
        for token, tag in sentence
    ]
    return metrics.classification_report(gold, pred, output_dict=output_dict)


def get_token_tag_tuples(sent):
    return [nltk.tag.str2tuple(t) for t in sent.split()]


def get_tagged_sentences(text):
    sentences = []

    blocks = text.split("======================================")
    for block in blocks:
        sents = block.split("\n\n")
        for sent in sents:
            sent = sent.replace("\n", "").replace("[", "").replace("]", "")
            if sent is not "":
                sentences.append(sent)
    return sentences


def load_treebank_splits(datadir):

    train = []
    dev = []
    test = []

    print("Loading treebank data...")

    for subdir, dirs, files in os.walk(datadir):
        for filename in files:
            if filename.endswith(".pos"):
                filepath = subdir + os.sep + filename
                with open(filepath, "r") as fh:
                    text = fh.read()
                    if int(subdir.split(os.sep)[-1]) in range(0, 19):
                        train += get_tagged_sentences(text)

                    if int(subdir.split(os.sep)[-1]) in range(19, 22):
                        dev += get_tagged_sentences(text)

                    if int(subdir.split(os.sep)[-1]) in range(22, 25):
                        test += get_tagged_sentences(text)

    print("Train set size: ", len(train))
    print("Dev set size: ", len(dev))
    print("Test set size: ", len(test))

    return train, dev, test


def main():

    # Set path for datadir
    datadir = os.path.join("data", "penn-treebank3-wsj", "wsj")

    train, dev, test = load_treebank_splits(datadir)

    ## For evaluation against the default NLTK POS tagger

    test_sentences = [get_token_tag_tuples(sent) for sent in test]
    tagged_test_sentences = [
        nltk.pos_tag([token for token, tag in sentence])
        for sentence in test_sentences
    ]
    evaluate(test_sentences, tagged_test_sentences)


if __name__ == "__main__":
    main()
