import gpu

COVID_PATH = {
    "train": "data/covid-qa/covid-qa-train.json",
    "dev": "data/covid-qa/covid-qa-dev.json",
    "test": "data/covid-qa/covid-qa-test.json",
}

DEVICE, GPU_ID = gpu.get_device()
MODEL = "deepset/roberta-base-squad2"
BATCH_SIZE = 8
MAX_LEN = 256  # The maximum length of a feature (question and context)
DOC_STRIDE = 128  # The authorized overlap between two part of the context when splitting it is needed.
IS_FINE_TUNE = True
NAME = f"{MODEL.split('/')[-1]}_{'finetune' if IS_FINE_TUNE else 'baseline'}"
