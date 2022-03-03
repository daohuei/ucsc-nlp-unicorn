from random import sample

from tqdm import tqdm

from feature import FeatureVector, Features, compute_features
from model import decode, backtrack
from data import sample_num

EARLY_STOP_NO_IMPROVE_LIMIT = 3


def optimizer(update_func="ssgd", l2_lambda=0.01):

    # only for adagrad
    # a feature vector for accumulated gradient square sum
    accum_sum = FeatureVector({})

    def adagrad(
        i,
        gradient,
        parameters,
        step_size,
    ):
        """
        AdaGrad update
        :param i: index of current instance
        :param gradient: func from index (int) in range(training_size) to a FeatureVector of the gradient
        :param parameters: FeatureVector.  Initial parameters.  Should be updated while training
        :param step_size: int. Learning rate, step size
        :return: updated parameters
        """
        # step_size / sqrt(accum_sum) * grad
        # accum_sum = sum_t(grad_t**2)
        grad = gradient(i)
        accum_sum.times_plus_equal(1, grad.square())
        parameters.times_plus_equal(
            -step_size, grad.divide(accum_sum.square_root())
        )
        return parameters

    def ssgd(i, gradient, parameters, step_size):
        """
        Stochastic sub-gradient descent update
        :param i: index of current instance
        :param gradient: func from index (int) in range(training_size) to a FeatureVector of the gradient
        :param parameters: FeatureVector.  Initial parameters.  Should be updated while training
        :param step_size: int. Learning rate, step size
        :return: updated parameters
        """
        # Look at the FeatureVector object.  You'll want to use the function times_plus_equal to update the parameters.
        # gradient in feature vector class
        grad = gradient(i)
        # w − α g(x, y)
        parameters.times_plus_equal(-step_size, grad)
        return parameters

    def l2_regularizer(i, gradient, parameters, step_size):
        """
        Stochastic sub-gradient descent update with L2 regularizer
        :param i: index of current instance
        :param gradient: func from index (int) in range(training_size) to a FeatureVector of the gradient
        :param parameters: FeatureVector.  Initial parameters.  Should be updated while training
        :param step_size: int. Learning rate, step size
        :return: updated parameters
        """
        # Look at the FeatureVector object.  You'll want to use the function times_plus_equal to update the parameters.
        # gradient in feature vector class
        grad = gradient(i)
        #  learning for l2 regularizer: w − α g(x, y) − αλw
        # λw
        regularizer = FeatureVector({})
        regularizer.times_plus_equal(l2_lambda, parameters)
        # w − α g(x, y) − αλw
        parameters.times_plus_equal(-step_size, grad)
        parameters.times_plus_equal(-step_size, regularizer)

    update = ssgd
    if update_func == "adagrad":
        update = adagrad
    elif update_func == "l2_regularizer":
        update = l2_regularizer

    def optimizer_func(
        training_size,
        epochs,
        gradient,
        parameters,
        training_observer,
        step_size=1,
    ):
        """
        Optimization Function (Based on Gradient Descent)
        :param training_size: int. Number of examples in the training set
        :param epochs: int. Number of epochs to run SGD for
        :param gradient: func from index (int) in range(training_size) to a FeatureVector of the gradient
        :param parameters: FeatureVector.  Initial parameters.  Should be updated while training
        :param training_observer: func that takes epoch and parameters.  You can call this function at the end of each
            epoch to evaluate on a dev set and write out the model parameters for early stopping.
        :param step_size: int. Learning rate, step size
        :return: final parameters
        """

        no_improve_count = 0

        best_params = parameters
        max_score = float("-inf")

        # go through every epochs
        for epoch in range(1, epochs + 1):
            # go through every training data
            for i in tqdm(range(training_size), desc="Training"):
                parameters = update(i, gradient, parameters, step_size)

            # dev score
            cur_score = training_observer(epoch, parameters)
            print(f"F-1 score at epoch {epoch}: {cur_score}")

            # updating best parameters
            if cur_score >= max_score:
                best_params = FeatureVector({})
                best_params.times_plus_equal(1, parameters)
                max_score = cur_score
                no_improve_count = 0
            else:
                # if no improvement
                no_improve_count += 1

            # if larger than tolerable no improvement times
            if no_improve_count > EARLY_STOP_NO_IMPROVE_LIMIT:
                # early stopping
                return best_params

        return best_params

    return optimizer_func


def hamming_loss(loss_val=10, penalty=0):
    """
    Modify the cost function to penalize mistakes three times more (penalty of 30) if the gold standard has a tag
    that is not O but the candidate tag is O.

    Args:
        penalty (Int)
    """

    def loss(gold, pred):
        result = loss_val
        if penalty > 0:
            if gold != "O" and pred == "O":
                result = penalty * result
        return result if gold != pred else 0

    return loss


def svm_with_cost_func(cost_func):
    def score(gold_labels, parameters, features):
        return svm_score(
            gold_labels, parameters, features, cost_func=cost_func
        )

    return score


def perceptron_score(gold_labels, parameters, features):
    # score function given current tag and previous tag with the parameter
    def score(cur_tag, pre_tag, i):
        # w dot f(x, y')
        return parameters.dot_product(
            features.compute_features(cur_tag, pre_tag, i)
        )

    return score


def svm_score(gold_labels, parameters, features, cost_func=hamming_loss()):
    # score function given current tag and previous tag with the parameter
    def score(cur_tag, pre_tag, i):
        # w dot f(x, y')
        cost_val = cost_func(gold_labels[i], cur_tag)
        cur_score = parameters.dot_product(
            features.compute_features(cur_tag, pre_tag, i)
        )
        return cur_score + cost_val

    return score


def get_gradient(data, feature_names, tagset, parameters, score_func):
    data = sample(data, sample_num)

    def subgradient(i):
        """
        Computes the subgradient of the Perceptron loss for example i
        :param i: Int
        :return: FeatureVector
        """
        # data point at i
        inputs = data[i]
        # get the token length
        input_len = len(inputs["tokens"])
        # get the gold labels
        gold_labels = inputs["gold_tags"]
        # get the features given feature names
        features = Features(inputs, feature_names)
        score = score_func(gold_labels, parameters, features)
        # use viterbi algorithm for decoding the tags
        tags = decode(input_len, tagset, score)
        # print(tags, gold_labels)
        # Add the predicted features
        fvector = compute_features(tags, input_len, features)

        # print("Input:", inputs)  # helpful for debugging
        # print("Predicted Feature Vector:", fvector.fdict)
        # print(
        #     "Predicted Score:", parameters.dot_product(fvector)
        # )  # compute_score(tags, input_len, score)

        # Subtract the features for the gold labels
        fvector.times_plus_equal(
            -1, compute_features(gold_labels, input_len, features)
        )
        # print(
        #     "Gold Labels Feature Vector: ",
        #     compute_features(gold_labels, input_len, features).fdict,
        # )
        # print(
        #     "Gold Labels Score:",
        #     parameters.dot_product(
        #         compute_features(gold_labels, input_len, features)
        #     ),
        # )
        # return the difference between features: which will be the update step
        return fvector

    return subgradient
