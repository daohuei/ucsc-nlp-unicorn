import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from data import tokenizer
from model import current_model
from config import DEVICE


def inference(model, loader):
    model.eval()

    output_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            X = batch["input_ids"]
            mask = batch["attention_mask"]
            input_batch = {
                "input_ids": X,
                "attention_mask": mask,
            }
            mask_token_index = (
                input_batch["input_ids"] == tokenizer.mask_token_id
            ).nonzero(as_tuple=True)[1]
            output = model(**input_batch, output_hidden_states=True)
            for batch_idx in range(output.logits.shape[0]):
                output_list.append(
                    output.logits[batch_idx, mask_token_index[batch_idx], :]
                )
    return output_list


def postprocess_output(output_token):
    output_word = tokenizer.decode(output_token)
    return output_word


def get_encoding_from_lm(model, word):
    tokenized_text = tokenizer(
        word, truncation=True, padding=True, return_tensors="pt"
    )
    output = model(**tokenized_text, output_hidden_states=True)
    cls_output = output.hidden_states[-1][0, 0, :]

    return cls_output


def predict_class(pred_encoding, class_encodings):
    return F.cosine_similarity(pred_encoding, class_encodings, dim=1).argmax(
        -1
    )

