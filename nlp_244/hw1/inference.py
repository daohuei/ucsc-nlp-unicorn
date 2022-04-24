import torch
from tqdm.auto import tqdm

from config import DEVICE, TASK
from data import slot_vocab, intent_vocab


def inference(model, loader):

    model.eval()

    output_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            X = batch["input_ids"]
            mask = batch["attention_mask"]

            outputs, output_val = None, None
            if TASK == "slot":
                outputs = model(**batch)
                output_val = outputs.logits
                for batch_idx in range(output_val.shape[0]):
                    output_list.append(output_val[batch_idx, :, :])
            elif TASK == "intent":
                outputs = model(X, mask)
                output_val = outputs
                for batch_idx in range(output_val.shape[0]):
                    output_list.append(output_val[batch_idx, :])

    # convert output to predictions
    test_predictions = []
    for test_output_val in output_list:
        test_pred = None
        if TASK == "intent":
            test_pred_binary = torch.where(test_output_val >= 0.5, 1, 0)
            test_pred = [
                intent_vocab.lookup_token(intent_idx)
                for intent_idx, val in enumerate(test_pred_binary)
                if val.item()
            ]
        elif TASK == "slot":
            test_pred_indexes = test_output_val.argmax(-1)
            test_pred = [
                slot_vocab.lookup_token(idx.item())
                for idx in test_pred_indexes
            ]
        assert test_pred != None
        test_predictions.append(test_pred)

    return test_predictions
