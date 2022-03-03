from data import dev_data, sample_num
from optimizer import *
from feature import Features, FeatureVector
from model import write_predictions
from evaluate import evaluate


def train(
    data,
    feature_names,
    tagset,
    epochs,
    optimizer,
    score_func=perceptron_score,
    step_size=1,
):
    """
    Trains the model on the data and returns the parameters
    :param data: Array of dictionaries representing the data.  One dictionary for each data point (as created by the
        make_data_point function).
    :param feature_names: Array of Strings.  The list of feature names.
    :param tagset: Array of Strings.  The list of tags.
    :param epochs: Int. The number of epochs to train
    :return: FeatureVector. The learned parameters.
    """

    parameters = FeatureVector({})  # creates a zero vector
    gradient = get_gradient(
        data, feature_names, tagset, parameters, score_func
    )

    def training_observer(epoch, parameters):
        """
        Evaluates the parameters on the development data, and writes out the parameters to a 'model.iter'+epoch and
        the predictions to 'ner.dev.out'+epoch.
        :param epoch: int.  The epoch
        :param parameters: Feature Vector.  The current parameters
        :return: Double. F1 on the development data
        """
        (_, _, f1) = evaluate(
            dev_data, parameters, feature_names, tagset, score_func
        )

        return f1

    # return the final parameters
    return optimizer(
        sample_num,
        epochs,
        gradient,
        parameters,
        training_observer,
        step_size=step_size,
    )
