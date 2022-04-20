import sys

import pandas as pd
from rouge import Rouge
import torch

from data import summary_max_len, word_vocab, get_data_loader, test_data
from helper import print_stage, write_predictions, write_scores
from model import Attention, Encoder, Decoder, Seq2Seq
from inference import inference
from constant import (
    ENC_EMB_DIM,
    ENC_HID_DIM,
    ENC_DROPOUT,
    DEC_EMB_DIM,
    DEC_HID_DIM,
    DEC_DROPOUT,
    DEVICE,
)

rouge = Rouge()

sys.setrecursionlimit(summary_max_len * summary_max_len + 10)


def calculate_rouges(preds, golds):

    result = {}
    scores = rouge.get_scores(preds, golds)
    score_df = pd.DataFrame(scores)
    for k in ["1", "2", "l"]:
        for m in ["p", "r", "f"]:
            key = f"rouge-{k}-{m}"
            value = (
                score_df[f"rouge-{k}"]
                .apply(lambda score_dict: score_dict[m])
                .mean()
            )
            result[key] = value

    return result


def evaluate_test(name):
    print_stage("Modeling")
    INPUT_DIM = len(word_vocab)
    OUTPUT_DIM = len(word_vocab)
    SRC_PAD_IDX = 0
    TRG_PAD_IDX = 0
    test_loader = get_data_loader(1, "test")
    test_golds = test_data.apply(lambda data: " ".join(data[1][1:])).tolist()

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(
        INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT
    )
    dec = Decoder(
        OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn
    )

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, DEVICE).to(DEVICE)

    print_stage("Evaluate Test Set")
    model.load_state_dict(torch.load(f"{name}-best-model.pt"))
    test_predictions = inference(model, test_loader, DEVICE)
    test_scores = calculate_rouges(test_predictions, test_golds)
    write_predictions(test_predictions, "test", name)
    write_scores(test_scores, "test", name)


if __name__ == "__main__":
    evaluate_test("seq2seq_batch_8_enc_emb_64_hid_128_dec_emb_64_hid_128")
    # rouge_score = calculate_rouge(test_data, SRC, TRG, model, device)

    # print("Rouge scores", np.array(rouge_score).mean(axis=0))
