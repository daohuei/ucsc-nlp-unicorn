"""classes for POS taggers"""
import numpy as np
from tqdm import tqdm

from utils import *


class BaselineTagger:
    def __init__(self):
        self.corpus = []  # training corpus

        self.most_frequent_table = (
            {}
        )  # key:val = word:most_freq_tag_of_such_word
        self.most_common_tag = (
            ""  # the most frequent tag no matter what word it is
        )

    def train(self, corpus):
        self.corpus = [preprocess_labeled_text(sent) for sent in corpus]

        word_tag_frequent_table = {}
        tag_counts = {}

        # calculate all the word:tag count and individual tag count
        for sent in self.corpus:
            for token in sent:
                word, tag = token
                tag_frequency = word_tag_frequent_table.get(word, {})
                tag_frequency[tag] = tag_frequency.get(tag, 0) + 1
                word_tag_frequent_table[word] = tag_frequency
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # calculate the most frequent tag for word
        for word, tag_freq in word_tag_frequent_table.items():
            self.most_frequent_table[word] = max(
                tag_freq.items(), key=lambda x: x[1]
            )[0]
        # calculate the most frequent tag among all tags
        self.most_common_tag = max(tag_counts.items(), key=lambda x: x[1])[0]

    def predict(self, corpus):
        tags = []
        for sent in corpus:
            tag_seq = []
            for word in sent:
                # if word in the dictionary, get the most frequent tag
                # if not just give most common tag
                tag = self.most_frequent_table.get(word, self.most_common_tag)
                tag_seq.append((word, tag))
            tags.append(tag_seq)
        return tags


class HMMTagger:
    def __init__(self):
        # preprocessed and labeled corpus for the POS task
        self.corpus = []
        # frequency of word tokens
        self.vocab = {}
        # all type of tags
        self.tag_list = []
        # table for the transition prob of all combination of tag->tag in the corpus
        self.transition_table = []
        # table for the emission prob of all combination of word:tag in the corpus
        self.emission_table = []
        # the emission probability for <UNK>:tag => actually it is just a prob for every single tag
        self.unknown_emission_prob = {}

    def train(self, corpus, alpha=1):
        self.corpus = [preprocess_labeled_text(sent) for sent in corpus]
        self.vocab = self.get_vocab(self.corpus)
        (
            self.transition_table,
            self.emission_table,
            self.unknown_emission_prob,
            self.tag_list,
        ) = self.build_tables(alpha)

        self.word2idx = {
            word: idx for idx, word in enumerate(self.vocab.keys())
        }
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.tag_list)}
        self.idx2word = {
            idx: word for idx, word in enumerate(self.vocab.keys())
        }
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.tag_list)}

    def build_tables(self, alpha=1):
        # key is prev tag and current tag, value is count
        transitions = self.get_transitions(self.corpus)
        # key is tag, token, value is count
        emissions = self.get_emissions(self.corpus)

        # store frequency of each tag
        tag_dict = self.get_tag_freq(self.corpus)
        tag_list = tag_dict.keys()

        transition_table = self.create_transition_table(
            transitions, tag_dict, tag_list, alpha
        )
        emission_table, unknown_emission_prob = self.create_emission_table(
            emissions, tag_dict, tag_list, self.vocab, alpha
        )

        return (
            transition_table,
            emission_table,
            unknown_emission_prob,
            tag_list,
        )

    def get_vocab(self, sents):
        vocab = {}
        for sent in sents:
            for token in sent:
                word = token[0]
                vocab[word] = vocab.get(word, 0) + 1
        return vocab

    def get_transitions(self, sents):
        transitions = {}
        for sent in sents:
            for i in range(1, len(sent)):
                bigram_tags = (sent[i - 1][1], sent[i][1])
                transitions[bigram_tags] = transitions.get(bigram_tags, 0) + 1
        return transitions

    def get_emissions(self, sents):
        emissions = {}
        for sent in sents:
            for token_tag_pair in sent:
                emissions[token_tag_pair] = (
                    emissions.get(token_tag_pair, 0) + 1
                )
        return emissions

    def get_tag_freq(self, sents):
        tag_dict = {}
        for sent in sents:
            for _, tag in sent:
                tag_dict[tag] = tag_dict.get(tag, 0) + 1
        return tag_dict

    def create_transition_table(self, transitions, tag_dict, tags, alpha=1):
        transition_table = []  # 2-dim list
        for prev_tag in tags:
            prob_list = []
            for current_tag in tags:
                prev_count = tag_dict.get(prev_tag, 0)
                bigram_count = transitions.get((prev_tag, current_tag), 0)
                prob = (bigram_count + alpha) / (
                    prev_count + (alpha * len(tags))
                )
                prob_list.append(np.log(prob))
            transition_table.append(prob_list)
        return transition_table

    def create_emission_table(self, emissions, tag_dict, tags, vocab, alpha):
        emission_table = []  # 2-dim list
        unknown_emission_prob = {}
        total_tag_counts = sum(tag_dict.values())
        for tag in tags:
            prob_list = []
            tag_count = tag_dict.get(tag, 0)
            for word in vocab.keys():
                word_tag_count = emissions.get((word, tag), 0)
                prob = (word_tag_count + alpha) / (
                    tag_count + (alpha * len(tags))
                )
                prob_list.append(np.log(prob))
            emission_table.append(prob_list)
            unknown_emission_prob[tag] = (tag_count + alpha) / (
                total_tag_counts + (alpha * len(tags))
            )
        return emission_table, unknown_emission_prob

    def viterbi_decode(self, sent):
        tags = []
        viterbi_matrix = []

        # Initial step
        initial = []  # empty array for start token
        viterbi_matrix.append(initial)
        first_token = sent[1]
        first_token_scores = []
        for i, tag in enumerate(self.tag_list):
            transition_prob = self.transition_table[self.tag2idx[START_TOKEN]][
                i
            ]
            emission_prob = self.unknown_emission_prob[tag]
            if first_token in self.word2idx.keys():
                emission_prob = self.emission_table[i][
                    self.word2idx[first_token]
                ]
            # calculate all the tag start from the start token
            first_token_scores.append(
                (self.tag2idx[START_TOKEN], transition_prob + emission_prob)
            )
        viterbi_matrix.append(first_token_scores)

        # recursive step
        for t, token in enumerate(sent):
            if t <= 1:
                continue
            max_scores = []
            for i, tag in enumerate(self.tag_list):
                max_score = float("-inf")
                candidate = None
                emission_prob = self.unknown_emission_prob[tag]
                if token in self.word2idx.keys():
                    emission_prob = self.emission_table[i][
                        self.word2idx[token]
                    ]
                # go through every previous score that already be calculated in the viterbi matrix
                for j, score in enumerate(viterbi_matrix[t - 1]):
                    _, prev_max_log_prob = score
                    transition_prob = self.transition_table[j][i]
                    new_score = (
                        emission_prob + transition_prob + prev_max_log_prob
                    )
                    if new_score > max_score:
                        max_score = new_score
                        candidate = j
                max_scores.append((candidate, max_score))
            viterbi_matrix.append(max_scores)

        # start with the stop tag
        max_tag = self.tag2idx[STOP_TOKEN]
        tags.append((STOP_TOKEN, self.idx2tag[max_tag]))

        # find best path in viterbi matrix
        for i in reversed(range(1, len(viterbi_matrix))):
            max_tag = viterbi_matrix[i][max_tag][0]
            tags.append((sent[i - 1], self.idx2tag[max_tag]))

        # since it is found backward, we need to reverse it
        tags.reverse()
        return tags

    def predict(self, corpus):
        all_tags = []
        for sent in tqdm(corpus):
            prediction_tags = self.viterbi_decode(sent)
            all_tags.append(prediction_tags)
        return all_tags
