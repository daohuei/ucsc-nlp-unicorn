from data import train_data, dev_data, test_data, tagset
from train import train
from model import write_predictions
from optimizer import *
from evaluate import *
from feature import feature_1_to_4, feature_full


def perceptron_experiment(feature_names, name, optimizer, write_params=False):
    print(f"=============Results of Experiment {name}============")
    parameters = train(
        train_data,
        feature_names,
        tagset,
        epochs=5,
        score_func=perceptron_score,
        optimizer=optimizer,
    )
    evaluate_dev_and_test(
        parameters, feature_names, perceptron_score, name=name
    )
    if write_params:
        # write parameters
        parameters.write_to_file(f"{name}.parameters")


def svm_experiment(cost, name):
    print(f"\n\n=============Results of Experiment {name}============\n\n")
    step_sizes = [0.1, 1, 10]
    l2_lambdas = [0.1, 1, 10]
    tunning_columns = ["step_size", "l2_lambda", "precision", "recall", "f1"]
    score = svm_with_cost_func(cost)
    max_f1 = 0
    best_parameters = None
    reports = []
    for step_size in step_sizes:
        for l2_lambda in l2_lambdas:
            print(f"step: {step_size}, lambda: {l2_lambda}")
            parameters = train(
                train_data,
                feature_full,
                tagset,
                epochs=5,
                step_size=step_size,
                score_func=score,
                optimizer=optimizer(
                    update_func="l2_regularizer", l2_lambda=l2_lambda
                ),
            )
            precision, recall, f1 = evaluate(
                dev_data, parameters, feature_full, tagset, score
            )
            reports.append([step_size, l2_lambda, precision, recall, f1])
            if f1 > max_f1:
                tuned_step_size = step_size
                tuned_l2_lambda = l2_lambda
                best_parameters = parameters
    print(f"\nBest!! step: {step_size}, lambda: {l2_lambda}\n")
    write_reports(reports, f"{name}.tuning.report", tunning_columns)
    evaluate_dev_and_test(best_parameters, feature_full, score, name=name)


def experiment():
    # TODO: average loss over epochs

    # feat1-4 perceptron ssgd
    perceptron_experiment(
        feature_names=feature_1_to_4,
        name="feat1-4_perceptron_ssgd",
        optimizer=optimizer(),
    )

    # feat-full perceptron ssgd
    perceptron_experiment(
        feature_names=feature_full,
        name="featfull_perceptron_ssgd",
        optimizer=optimizer(),
        write_params=True,
    )

    # feat-full perceptron adagrad
    perceptron_experiment(
        feature_names=feature_full,
        name="featfull_perceptron_adagrad",
        optimizer=optimizer(update_func="adagrad"),
    )

    # feat-full svm ssgd: tune step_size and regularizer
    svm_experiment(hamming_loss(), name="feat-full_svm_ssgd")

    # feat-full modified-svm ssgd: tune step_size and regularizer
    svm_experiment(
        hamming_loss(penalty=30), name="feat-full_modified-svm_ssgd"
    )


def evaluate_dev_and_test(parameters, feature_names, score, name):
    report_cols = ["precision", "recall", "f1"]

    # generating dev report
    report = evaluate(dev_data, parameters, feature_names, tagset, score)
    dev_precision, dev_recall, dev_f1 = report
    print(
        f"\nDev=> Precision: {dev_precision} Recall: {dev_recall} F-1: {dev_f1}"
    )
    write_predictions(
        f"{name}.dev.pred",
        dev_data,
        parameters,
        feature_names,
        tagset,
        score,
    )
    write_reports(
        [list(report)],
        f"{name}.dev.report",
        report_cols,
    )

    # generating test report
    report = evaluate(test_data, parameters, feature_names, tagset, score)
    test_precision, test_recall, test_f1 = report
    print(
        f"Test=> Precision: {test_precision} Recall: {test_recall} F-1: {test_f1}\n"
    )
    write_predictions(
        f"{name}.test.pred",
        test_data,
        parameters,
        feature_names,
        tagset,
        score,
    )
    write_reports(
        [list(report)],
        f"{name}.test.report",
        report_cols,
    )
