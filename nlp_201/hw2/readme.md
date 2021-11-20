# NLP 201 Homework 2

## Prerequisite

Vanilla Python3

## Usage

### Run with argument

```bash
python3 main.py [debug|regular|additive|interpolation] ngram train_portion freq_restriction [hyperparameter:alpha(if additive)|l1 l2 l3(if interpolation)]
```

-   debug: for debugging the model given the sample mentioned in the problem.
-   regular: without any smoothing
-   additive: additive smoothing
-   interpolation: linear interpolation
-   ngram: 1:unigram, 2:bigram, 3:trigram
-   train_portion: [0-1] the portion of training corpus will be used to train the model
-   freq_restriction: use to replace the low frequency word by <UNK> token
-   alpha: for the additive smoothing
-   l1,l2,l3: lambdas for linear interpolation

### Run Experiment

```bash
python3 experiments.py
```

This command will run all the experiment mentioned in the problem set.

## Python File

1. `tokens.py`: contains special token constants
2. `utils.py`: contains utilities of text preprocessing
3. `ngram.py`: the main logic of the ngram language model, only include unigram, bigram, trigram
4. `train.py`: function to train the ngram, defined training functions with different smoothing methods
5. `main.py`: a script that can train the model with arguments, see usage section
6. `experiments.py`: a script for running different experiments that mentioned in the problem set of homework 2
