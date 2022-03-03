from constants import START, STOP
from feature import Features


def backtrack(viterbi_matrix, tagset, max_tag):
    tags = []
    for k in reversed(range(len(viterbi_matrix))):
        last_max_tag_idx = tagset.index(max_tag)
        viterbi_list = viterbi_matrix[k]
        max_tag, _ = viterbi_list[last_max_tag_idx]
        tags = [max_tag] + tags
    return tags


def decode(input_len, tagset, score_func):
    """Viterbi Decoding

    Args:
        input_len (int): input length
        tagset (list): the list of all tags
        score_func (func): score function

    Returns:
        tags (list): predicted tag sequence
    """
    tags = []
    viterbi_matrix = []

    # initial step
    initial_list = []
    for tag in tagset:
        score = score_func(tag, START, 1)
        initial_list.append((START, score))
    viterbi_matrix.append(initial_list)

    # recursion step
    for t in range(2, input_len - 1):

        viterbi_list = []
        for tag in tagset:
            # max() and argmax()
            max_tag = None
            max_score = float("-inf")
            for prev_tag in tagset:
                last_viterbi_list = viterbi_matrix[t - 2]
                prev_tag_idx = tagset.index(prev_tag)
                last_score = last_viterbi_list[prev_tag_idx][1]
                score = score_func(tag, prev_tag, t) + last_score
                if score > max_score:
                    max_score = score
                    max_tag = prev_tag
            viterbi_list.append((max_tag, score))
        viterbi_matrix.append(viterbi_list)
    # termination step
    tags = [STOP] + tags

    # calculate the max tag
    last_viterbi_list = []
    for tag in tagset:
        stop_score = score_func(STOP, tag, input_len - 1)
        prev_score = viterbi_matrix[-1][tagset.index(tag)][1]
        score = stop_score + prev_score
        last_viterbi_list.append((tag, score))
    max_tag, _ = max(last_viterbi_list, key=lambda tuple: tuple[1])

    tags = backtrack(viterbi_matrix, tagset, max_tag) + [max_tag] + tags
    return tags


def predict(inputs, input_len, parameters, feature_names, tagset, score_func):
    """

    :param inputs:
    :param input_len:
    :param parameters:
    :param feature_names:
    :param tagset:
    :return:
    """
    features = Features(inputs, feature_names)
    gold_labels = inputs["gold_tags"]

    score = score_func(gold_labels, parameters, features)

    return decode(input_len, tagset, score)


def write_predictions(
    out_filename, all_inputs, parameters, feature_names, tagset, score_func
):
    """
    Writes the predictions on all_inputs to out_filename, in CoNLL 2003 evaluation format.
    Each line is token, pos, NP_chuck_tag, gold_tag, predicted_tag (separated by spaces)
    Sentences are separated by a newline
    The file can be evaluated using the command: python conlleval.py < out_file
    :param out_filename: filename of the output
    :param all_inputs:
    :param parameters:
    :param feature_names:
    :param tagset:
    :return:
    """
    with open(out_filename, "w", encoding="utf-8") as f:
        for inputs in all_inputs:
            input_len = len(inputs["tokens"])
            tag_seq = predict(
                inputs,
                input_len,
                parameters,
                feature_names,
                tagset,
                score_func,
            )
            for i, tag in enumerate(
                tag_seq[1:-1]
            ):  # deletes <START> and <STOP>
                f.write(
                    " ".join(
                        [
                            inputs["tokens"][i + 1],
                            inputs["pos"][i + 1],
                            inputs["NP_chunk"][i + 1],
                            inputs["gold_tags"][i + 1],
                            tag,
                        ]
                    )
                    + "\n"
                )  # i + 1 because of <START>
            f.write("\n")


def compute_score(tag_seq, input_length, score):
    """
    Computes the total score of a tag sequence
    :param tag_seq: Array of String of length input_length. The tag sequence including <START> and <STOP>
    :param input_length: Int. input length including the padding <START> and <STOP>
    :param score: function from current_tag (string), previous_tag (string), i (int) to the score.  i=0 points to
        <START> and i=1 points to the first token. i=input_length-1 points to <STOP>
    :return:
    """
    total_score = 0
    for i in range(1, input_length):
        total_score += score(tag_seq[i], tag_seq[i - 1], i)
    return total_score


def test_decoder():
    # See https://classes.soe.ucsc.edu/nlp202/Winter21/assignments/A1_Debug_Example.pdf

    tagset = ["NN", "VB"]  # make up our own tagset

    def score_wrap(cur_tag, pre_tag, i):
        retval = score(cur_tag, pre_tag, i)
        print(
            "Score("
            + cur_tag
            + ","
            + pre_tag
            + ","
            + str(i)
            + ") returning "
            + str(retval)
        )
        return retval

    def score(cur_tag, pre_tag, i):
        if i == 0:
            print(
                "ERROR: Don't call score for i = 0 (that points to <START>, with nothing before it)"
            )
        if i == 1:
            if pre_tag != "<START>":
                print(
                    "ERROR: Previous tag should be <START> for i = 1. Previous tag = "
                    + pre_tag
                )
            if cur_tag == "NN":
                return 6
            if cur_tag == "VB":
                return 4
        if i == 2:
            if cur_tag == "NN" and pre_tag == "NN":
                return 4
            if cur_tag == "NN" and pre_tag == "VB":
                return 9
            if cur_tag == "VB" and pre_tag == "NN":
                return 5
            if cur_tag == "VB" and pre_tag == "VB":
                return 0
        if i == 3:
            if cur_tag != "<STOP>":
                print(
                    "ERROR: Current tag at i = 3 should be <STOP>. Current tag = "
                    + cur_tag
                )
            if pre_tag == "NN":
                return 1
            if pre_tag == "VB":
                return 1

    predicted_tag_seq = decode(4, tagset, score_wrap)
    print("Predicted tag sequence should be = <START> VB NN <STOP>")
    print("Predicted tag sequence = " + " ".join(predicted_tag_seq))
    print(
        "Score of ['<START>','VB','NN','<STOP>'] = "
        + str(compute_score(["<START>", "VB", "NN", "<STOP>"], 4, score))
    )
    print("Max score should be = 14")
    print("Max score = " + str(compute_score(predicted_tag_seq, 4, score)))
