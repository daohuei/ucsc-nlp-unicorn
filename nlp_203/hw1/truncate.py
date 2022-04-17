import spacy
import pandas as pd
from tqdm import tqdm

from helper import read_files, print_stage
from constant import SRC_FILENAME

spacy_en = spacy.load("en_core_web_sm")


def truncate(tokens, size):
    if size <= 0:
        return []
    if len(tokens) <= size:
        return tokens

    to_remove_count = int((len(tokens) - size) / 2)
    return tokens[to_remove_count:-to_remove_count]


def apply_truncate(row, truncate_size):
    text = list(row["text"])
    if row["text_len"] <= truncate_size:
        return text
    clean_text = row["clean_text"]
    clean_text_len = row["clean_text_len"]
    if clean_text_len <= truncate_size:
        return clean_text
    sents = row["sents"]
    truncated_text = []
    for sent in sents:
        tokens = [token for token in list(sent) if is_clean(token)]
        size = round(truncate_size * len(tokens) / clean_text_len)
        truncated_text += truncate(tokens, size)

    return truncated_text


def is_clean(spacy_token):
    return (
        not spacy_token.is_stop
        and not spacy_token.is_punct
        and len(str(spacy_token).strip()) > 0
    )


def apply_spacy(df):
    results = []
    for text in tqdm(df["text"]):
        spacy_text = spacy_en(text)
        results += [spacy_text]
    return results


def get_truncated_df(df):
    print_stage("applying spacy")
    df["text"] = apply_spacy(df)
    # original text length
    df["text_len"] = df["text"].apply(len)

    print_stage("cleaning text")
    # text without stopwords and punctuation
    df["clean_text"] = df["text"].apply(
        lambda doc: [token for token in list(doc) if is_clean(token)]
    )
    df["clean_text_len"] = df["clean_text"].apply(len)

    print_stage("sentencizing")
    # sentencize the text
    df["sents"] = df["text"].apply(lambda doc: list(doc.sents))

    print_stage("truncating")
    # truncate size will be the average of clean text length
    average_text_len = int(df["clean_text_len"].median())
    df["truncated_text"] = df.apply(
        lambda row: apply_truncate(row, average_text_len), axis=1
    )
    df["truncated_text_len"] = df["truncated_text"].apply(len)
    df["truncated_full_text"] = df["truncated_text"].apply(
        lambda tokens: " ".join([str(token) for token in list(tokens)])
    )
    return df


def write_truncated_train_file(df):
    print_stage("writing to file")
    file_name = "cnndm/data/truncated_train.txt.src"
    with open(file_name, "w") as f:
        for text in tqdm(df):
            f.write(f"{text}\n")


def generate_truncated_dataset():
    inputs = read_files(SRC_FILENAME["train"])
    df = pd.DataFrame({"text": inputs})
    df = get_truncated_df(df)
    write_truncated_train_file(df["truncated_full_text"])


if __name__ == "__main__":
    generate_truncated_dataset()