import numpy as np

word_tag_prob = {}
tag_bigram_prob = {}
max_score_dict = {}
tag_set = set()


def score(words, i, cur_tag, prev_tag):
    return np.log(word_tag_prob[(words[i], cur_tag)]) + np.log(
        tag_bigram_prob[(cur_tag, prev_tag)]
    )


# this will recur n times => which the time complexity will be O(n * |T|) with memorization of O(n * |T|)
# if there is no memorization, it will be O(|T| ^ n)
def max_score_tag(words, n, cur_tag, max_score_dict):
    if n == 1:
        # if it is the first tag, simply calculate the score and return empty candidates
        return [], score(words, n, cur_tag, "<START>")
    max_score = float("-inf")
    candidate_tag = None
    # go through every tags: time complexity: O(|T|), the size of the tag_set
    for prev_tag in tag_set:
        candidate_tag_sequences, prev_score = [], float("-inf")
        if (prev_tag, n - 1) in max_score_dict.keys():
            # if the max score of previous tag already been calculated
            candidate_tag_sequences, prev_score = max_score_dict[
                (prev_tag, n - 1)
            ]
        else:
            # get the best sequence candidates of given tag and index(sequence length)
            candidate_tag_sequences, prev_score = max_score_tag(
                words, n - 1, prev_tag, max_score_dict
            )
        # calculate current score and sum up with previous score
        cur_score = prev_score + score(words, n, cur_tag, prev_tag)
        # if it becomes the maximum
        if cur_score > max_score:
            # update max score and candidate tags
            max_score = cur_score
            candidate_tag = prev_tag
    # store the calculation in the dictionary
    candidate_tag_sequences.append(candidate_tag)
    max_score_dict[(cur_tag, n)] = (candidate_tag_sequences, max_score)
    return candidate_tag_sequences, max_score


def add_semiring(a, b):
    return b if a[1] < b[1] else a


def mul_semiring(a, b):
    # since it is in the log space, so we use a + b
    return a + b


# this will recur n times => which the time complexity will be O(n * |T|) with memorization of O(n * |T| ^ 2)
# if there is no memorization, it will be O(|T| ^ 2n)
def max_score_tag_semiring(words, n, cur_tag, max_score_dict):
    if n == 1:
        # if it is the first tag, simply calculate the score and return empty candidates
        return [], score(words, n, cur_tag, "<START>")
    max_score = (None, float("-inf"))
    # go through every tags: time complexity: O(|T|), the size of the tag_set
    for prev_tag in tag_set:
        candidate_tag_sequences, prev_score = [], float("-inf")
        if (prev_tag, n - 1) in max_score_dict.keys():
            # if the max score of previous tag already been calculated
            candidate_tag_sequences, prev_score = max_score_dict[
                (prev_tag, n - 1)
            ]
        else:
            # get the best sequence candidates of given tag and index(sequence length)
            candidate_tag_sequences, prev_score = max_score_tag(
                words, n - 1, prev_tag, max_score_dict
            )
        # calculate current score and sum up with previous score
        cur_score = (
            prev_tag,
            mul_semiring(prev_score, score(words, n, cur_tag, prev_tag)),
        )
        max_score = add_semiring(max_score, cur_score)
    # store the calculation in the dictionary
    candidate_tag_sequences.append(max_score[0])
    max_score_dict[(cur_tag, n)] = (candidate_tag_sequences, max_score[1])
    return candidate_tag_sequences, max_score[1]
