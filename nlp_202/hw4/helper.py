import torch


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    """return indexes sequence of input word sequence

    Args:
        seq (list): word sequence
        to_ix (dict): word to index map

    Returns:
        Tensor: indexes in Tensor format
    """
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec):
    """Compute log sum exp in a numerically stable way for the forward algorithm

    Args:
        vec (Tensor): score vector

    Returns:
        Tensor: log sum exp
    """
    # TODO: not sure what this function is doing
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(
        torch.sum(torch.exp(vec - max_score_broadcast))
    )


def unpad_sequence(sequences, seq_lens):
    results = []
    for i, seq in enumerate(sequences):
        results += [seq[: seq_lens[i]]]
    return results


def convert_batch_sequence(batch_sequence, vocab):
    return [convert_sequence(sequence, vocab) for sequence in batch_sequence]


def convert_sequence(sequence, vocab):
    return [vocab.idx2token[idx] for idx in sequence]


def convert_to_char_tensor(token_vector, word_vocab, char_vocab, max_word_len):
    char_tensor = []
    for idx in token_vector:
        word = word_vocab.lookup_token(idx.item())
        padded_char = word_to_padded_char(word, char_vocab, max_word_len)
        char_tensor.append(padded_char)
    return torch.cat(char_tensor, 0)


def word_to_padded_char(word, char_vocab, max_word_len):
    processed_chars = [char_vocab.lookup_index(c) for c in word]
    processed_chars = padding_char(processed_chars, max_word_len)
    # batch * channel * sequence length
    processed_chars = torch.tensor(
        processed_chars, dtype=torch.long
    ).unsqueeze(0)
    return processed_chars


def padding_char(char, max_len):
    while len(char) < max_len:
        char = [0] + char
        if len(char) == max_len:
            break
        char = char + [0]
    return char