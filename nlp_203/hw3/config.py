import gpu

COVID_PATH = {
    "train": "data/covid-qa/covid-qa-train.json",
    "dev": "data/covid-qa/covid-qa-dev.json",
    "test": "data/covid-qa/covid-qa-test.json",
}

DEVICE, GPU_ID = gpu.get_device()
MODEL = "deepset/roberta-base-squad2"
ADAPTER = "AdapterHub/roberta-base-pf-squad_v2"

# MODEL = "roberta-base-squad2_finetune/checkpoint-3108"
# ADAPTER = "roberta-base-squad2_finetune_adapter/checkpoint-15540/squad_v2"

BATCH_SIZE = 8
MAX_LEN = 385  # The maximum length of a feature (question and context)
DOC_STRIDE = 128  # The authorized overlap between two part of the context when splitting it is needed.
EPOCHS = 3
IS_FINE_TUNE = False
IS_ADAPTER = True
NAME = f"{MODEL.split('/')[-1]}_{'finetune' if IS_FINE_TUNE else 'baseline'}{'_adapter_squad_v2' if IS_ADAPTER else ''}_{EPOCHS}"
