# NLP 202 Homework 3

## Prerequisite

-   Python 3.8.2
-   [tqdm](https://tqdm.github.io/) Python module for visible progress bar
-   [PyTorch](https://pytorch.org) Deep Learning Framework

## Files

-   `A4-data`: folder of data for evaluating and training (and testing)
-   `conlleval.py` : eval function for f-1 score/precision/recall that provided by previous homework
-   `constants.py`: store constants such as start and stop token
-   `data.py`: include read data function and data loading function
-   `evaluate.py`: provide evaluation method and write the report to files
-   `experiment.py`: all experiments setup and execution are defined in this script
-   `gpu.py`: a script that can automatically choose most idle GPU for PyTorch training
-   `main.py`: just the main script, which can be access through command line
-   `model.py`: code for BiLSTM model, include decoding algorithm, Char-level CNN, and loss function
-   `starter_code.py`: Same as the starter code in [here](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#dynamic-versus-static-deep-learning-toolkits)
-   `train.py`: Training function including implementations of looping over training steps and early stopping

## Usage

The first argument is for the experiment:

```bash
python main.py --help

Usage: python main.py [OPTIONS] COMMAND [ARGS]...

Commands:
  bi-lstm-crf
  bi-lstm-crf-char-cnn
  bi-lstm-crf-ramp-loss
  bi-lstm-crf-soft-ramp-loss
  bi-lstm-crf-softmax-margin-loss
  bi-lstm-crf-svm-loss

```

For the rest of arguments, you can also use help given after experiment argument:

```bash
python main.py bi-lstm-crf --help

Usage: python main.py bi-lstm-crf [OPTIONS] NAME

Arguments:
  NAME  [required]

Options:
  --emb-dim INTEGER       [default: 5]
  --hidden-dim INTEGER    [default: 4]
  --epoch-num INTEGER     [default: 2]
  --batch-size INTEGER    [default: 2]
  --lr FLOAT              [default: 0.01]
  --lamb FLOAT            [default: 0.0001]
  --resume / --no-resume  [default: False]
  --cost-val INTEGER      [default: 10]
  --help                  Show this message and exit.
```

Every argument has its own default value except that the name is required.

Example usage:

```bash
python main.py bi-lstm-crf my-experiment-name
```

## Output Format

```
{experiment}.{pred|report|time|pt|hp}
```

-   experiment: The experiment name you give to the script
-   output type:
    -   `pred`: prediction results
    -   `report`: classification report: precision, recall, f1
    -   `time`: running time given batch
    -   `pt`: learned parameters for the model, can be loaded by PyTorch given matched NN module
    -   `hp`: Hyperparameters for the experiment
