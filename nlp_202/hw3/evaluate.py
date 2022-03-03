import pandas as pd
from tqdm import tqdm

from conlleval import evaluate as conllevaluate
from model import predict


def evaluate(
    data, parameters, feature_names, tagset, score_func, verbose=False
):
    """
    Evaluates precision, recall, and F1 of the tagger compared to the gold standard in the data
    :param data: Array of dictionaries representing the data.  One dictionary for each data point (as created by the
        make_data_point function)
    :param parameters: FeatureVector.  The model parameters
    :param feature_names: Array of Strings.  The list of features.
    :param tagset: Array of Strings.  The list of tags.
    :return: Tuple of (prec, rec, f1)
    """
    all_gold_tags = []
    all_predicted_tags = []
    for inputs in tqdm(data, desc="Evaluating"):
        all_gold_tags.extend(
            inputs["gold_tags"][1:-1]
        )  # deletes <START> and <STOP>
        input_len = len(inputs["tokens"])
        all_predicted_tags.extend(
            predict(
                inputs,
                input_len,
                parameters,
                feature_names,
                tagset,
                score_func,
            )[1:-1]
        )  # deletes <START> and <STOP>
    return conllevaluate(all_gold_tags, all_predicted_tags, verbose)


def write_reports(reports, filename, columns):
    report_map = {}
    for i, column in enumerate(columns):
        values = []
        for report in reports:
            value = report[i]
            values.append(value)
        report_map[column] = values

    report_df = pd.DataFrame(data=report_map)
    report_df.to_csv(filename, index=False)
