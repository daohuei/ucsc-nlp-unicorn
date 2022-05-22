COVID_PATH = {
    "train": "data/covid-qa/covid-qa-train.json",
    "dev": "data/covid-qa/covid-qa-dev.json",
    "test": "data/covid-qa/covid-qa-test.json",
}

MODEL = "deepset/roberta-base-squad2"
BATCH_SIZE = 16
MAX_LEN = 384  # The maximum length of a feature (question and context)
DOC_STRIDE = 128  # The authorized overlap between two part of the context when splitting it is needed.
NAME = f"{MODEL.split('/')[-1]}_baseline"
