import torch
from tqdm import tqdm

from constant import START, STOP
from data import word_vocab


def inference(model, loader, device):

    model.eval()

    preds = []
    for batch in tqdm(loader):
        text, _, text_len, _ = batch
        text = text = text.view(-1, 1)
        pred = summarize(text, text_len, model, device)
        preds.append(" ".join(pred))

    return preds


def summarize(text_tensor, lens, model, device, max_len=50):
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(text_tensor, lens)

    mask = model.create_mask(text_tensor)

    summary_idxes = [word_vocab.lookup_index(START)]

    attentions = torch.zeros(max_len, 1, len(text_tensor)).to(device)

    for i in range(max_len):
        last_output = torch.LongTensor([summary_idxes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, attention = model.decoder(
                last_output, hidden, encoder_outputs, mask
            )

        attentions[i] = attention

        pred_token = output.argmax(1).item()
        summary_idxes.append(pred_token)

        if pred_token == word_vocab.lookup_index(STOP):
            break

    summary_tokens = [word_vocab.lookup_token(idx) for idx in summary_idxes]

    return summary_tokens
