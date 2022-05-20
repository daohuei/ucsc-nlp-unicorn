import torch
from tqdm.auto import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from data import get_data_loader, train_set, dev_set, test_set
from model import current_model
from helper import print_stage, dict_to_json
from inference import (
    inference,
    postprocess_output,
    get_encoding_from_lm,
    predict_class,
)
from config import NAME, PRE_PROMPT, SUF_PROMT


def experiment(name, pre_prompt, suf_prompt, split="dev"):
    assert split in ["train", "dev", "test"]
    dataset_map = {"train": train_set, "dev": dev_set, "test": test_set}
    data_loader = get_data_loader(split, pre_prompt, suf_prompt,)
    model = current_model

    print_stage("inference")
    output_list = inference(model, data_loader)

    print_stage("post-processing")
    output_word_list = []
    for logit in tqdm(output_list):
        output_token = logit.argmax(-1)
        output_word_list.append(postprocess_output(output_token))

    class_mapping = ["sad", "joy", "love", "anger", "fear", "surprise"]
    emotion_encodings = []
    for emotion in class_mapping:
        emotion_encoding = get_encoding_from_lm(model, emotion)
        emotion_encodings.append(emotion_encoding)
    emotion_encodings = torch.stack(emotion_encodings)

    print_stage("verbalizing")
    predictions = []
    for pred_word in tqdm(output_word_list):
        pred_encoding = get_encoding_from_lm(model, pred_word)
        predictions.append(
            predict_class(pred_encoding, emotion_encodings).item()
        )

    print_stage("evaluation")

    golds = dataset_map[split]["label"]
    report = classification_report(
        golds, predictions, output_dict=True, target_names=class_mapping
    )
    dict_to_json(report, f"{name}_{split}_report.json")
    c_matrix = confusion_matrix(golds, predictions)
    ConfusionMatrixDisplay(
        confusion_matrix=c_matrix, display_labels=class_mapping
    ).plot().figure_.savefig(f"{name}_{split}_c_matrix.jpg", dpi=150)


if __name__ == "__main__":
    # experiment(NAME, PRE_PROMPT, SUF_PROMT, "dev")
    experiment(NAME, PRE_PROMPT, SUF_PROMT, "test")
