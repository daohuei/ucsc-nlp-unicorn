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