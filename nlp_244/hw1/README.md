# NLP 244 Homework 1

## Prerequisite

-   Python 3.8.2
-   [tqdm](https://tqdm.github.io/) Python module for visible progress bar
-   [PyTorch](https://pytorch.org) Deep Learning Framework
-   [HuggingFace](https://huggingface.co/) Framework for large pre-trained transformer

You will also need to install the following python library:

```bash
python -m pip install sentencepiece datasets openpyxl
```

-   `sentencepiece` will be needed for the BERT-based tokenizer in HuggingFace
-   `datasets` is the HuggingFace Dataset library
-   `openpyxl` will be needed for ALBERT model

## Files

-   `auto_evaluation.py`: the same script as the one provided in the data folder
-   `config.py`: store constants and hyperparameters setup, you can modify the value in this script for testing different experiment setup
-   `data.py`: include data loading, splitting data, balancing data, and tokenization method
-   `gpu.py`: a script that can automatically choose the most idle GPU for PyTorch training
-   `helper.py`: other helper function like write report/prediction to files
-   `inference.py`: contain inference method
-   `model.py`: code of DistilBERT/ALBERT that to be fine-tuned on slot tagging/intent detection
-   `movie_dataset_eda.ipynb`: investigation on provided dataset for exploratory data analysis
-   `plot.py`: code for plotting loss/score per epoch
-   `train.py`: Training function including implementations of looping over training steps
-   `vocab.py`: contain vocabulary class, only used for slot/intent vocabulary in this work

## Usage

> You can modify hyperparameter in `config.py`

```bash
python train.py
```

## Output Format

```
{experiment}.{pred|score}
```

-   output type:
    -   `pred`: prediction results
    -   `score`: f1 report in CSV file f1
