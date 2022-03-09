# NLP 202 Homework 3

## Prerequisite

-   Python 3.8.2
-   [tqdm](https://tqdm.github.io/) Python module
-   [Pandas](https://pandas.pydata.org/docs/index.html)

## Files

-   `conlleval.py` : eval function for f-1 score/precision/recall that provided by starter code
-   `constants.py`: store constants for start and stop token
-   `data.py`: include read data function and load the data as constants
-   `evaluate.py`: provide evaluation method and write the report to files
-   `experiment.py`: all experiments setup and execution are defined in this script
-   `feature.py`: include class for feature and feature vector and also the functions that compute/construct features
-   `gazetteer.txt`: gazetteer list
-   `main.py`: just the main script
-   `model.py`: include decoding algorithm and predict function
-   `ner.{train|dev|test}`: CoNLL dataset
-   `optimizer.py`: include different optimizer(sgd, adagrad) and different score function(svm/perceptron score functions)
-   `train.py`: just a interface that call optimizer and score function from `optimizer.py`

## Output Format

```
{experiment}.{dev|test|tuning}.{pred|report|parameters}
```

-   experiments:
    -   `feat1-4_perceptron_ssgd`
    -   `featfull_perceptron_ssgd`
    -   `featfull_perceptron_adagrad`
    -   `feat-full_svm_ssgd`
    -   `feat-full_modified-svm_ssgd`
-   dataset:
    -   `dev`: prediction or report on dev set
    -   `test`: prediction or report on dev set
    -   `tuning`: report on hyperparameter tuning
-   output type:
    -   `pred`: prediction results
    -   `report`: classification report: precision, recall, f1
    -   `parameters`: learned features and weights, in the format of `feature_string feature_weight`

## Usage

### Run All Experiments

```bash
python3 main.py
```
