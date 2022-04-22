# NLP 204 Homework 1

## Prerequisite

-   Python 3.8.2
-   [tqdm](https://tqdm.github.io/) Python module for visible progress bar
-   [PyTorch](https://pytorch.org) Deep Learning Framework

## Files

-   `constant.py`: store constants such as start and stop token and hyperparameters
-   `data.py`: include read data function and data loading function, and vocabulary class
-   `evaluate.py`: provide evaluation method of ROUGE
-   `gpu.py`: a script that can automatically choose most idle GPU for PyTorch training
-   `helper.py`: other helper function like write report/prediction to files
-   `inference.py`: contain inference method like generating summary for prediction
-   `model.py`: code for Seq2Seq Attention Model
-   `plot.py`: code for plotting loss/PPL and epoch relations, and ROUGE and epoch relation
-   `starter_code.ipynb`: starter code provided by the assignment
-   `train.py`: Training function including implementations of looping over training steps
-   `truncate.py`: Code for sentence level truncating

## Usage

> You can modify hyperparameter in `constant.py`

```bash
python train.py
```

## Output Format

```
{experiment}.{pred|score}
```

-   output type:
    -   `pred`: prediction results
    -   `score`: ROUGE report in CSV file: precision, recall, f1
