import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from constants import START_TAG, STOP_TAG, DEVICE
from helper import argmax, log_sum_exp, hamming_loss, convert_to_char_tensor
from data import tag_vocab, max_word_len, char_vocab, word_vocab


class BiLSTM_CRF(nn.Module):
    def __init__(
        self,
        vocab_size,
        tag_to_ix,
        embedding_dim,
        hidden_dim,
        char_cnn=False,
        char_cnn_stride=2,
        char_cnn_kernel=2,
        char_embedding_dim=4,
        loss="crf_loss",
        cost=hamming_loss(),
    ):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.char_cnn = char_cnn
        self.max_word_len = max_word_len
        self.loss_type = loss
        self.cost = cost

        self.word_embeds = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )

        self.char_cnn_layer = CharCNN(
            max_word_len=max_word_len,
            embedding_dim=char_embedding_dim,
            kernel=char_cnn_kernel,
            stride=char_cnn_stride,
        )
        self.lstm_input_dim = embedding_dim
        if char_cnn:
            self.lstm_input_dim = (
                self.embedding_dim + self.char_cnn_layer.embedding_dim
            )

        self.lstm = nn.LSTM(
            self.lstm_input_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)
        )

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def init_hidden(self, batch):
        # cell state and hidden state initialization
        # D*num_layers x batch x hidden_dim
        # D = 2 if bidirectional=True otherwise 1
        return (
            torch.randn(2, batch, self.hidden_dim // 2).to(DEVICE),
            torch.randn(2, batch, self.hidden_dim // 2).to(DEVICE),
        )

    def _forward_alg(self, feats, golds=None, cost=None):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.0).to(
            DEVICE
        )  # 1 x |tag_set|
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.0

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence: the emission scores
        for i, feat in enumerate(feats):
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = (
                    feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                )
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = None
                if cost is not None:
                    # generate log sum exp(score + cost)
                    next_tag_var = (
                        forward_var
                        + trans_score
                        + emit_score
                        + cost(golds[i], next_tag)
                    )
                else:
                    next_tag_var = forward_var + trans_score + emit_score
                assert next_tag_var != None
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentences, seq_lens):
        # for getting sentence features from LSTM in tag space
        batch_size = len(sentences)

        self.hidden = self.init_hidden(batch=batch_size)
        # embeds shape: batch x seq_len  x emb_dim
        embeds = self.word_embeds(sentences)

        # character-level embedding
        if self.char_cnn:
            # generate char-level embedding for each token, go over sequence
            char_embeddeds = []
            for i in range(sentences.size()[1]):
                token_vector = sentences[:, i]
                char_tensor = convert_to_char_tensor(
                    token_vector, word_vocab, char_vocab, self.max_word_len
                ).to(DEVICE)
                char_embedded = self.char_cnn_layer(char_tensor)
                char_embedded = torch.transpose(char_embedded, 1, 2)
                char_embeddeds.append(char_embedded)
            # concatenate all chars together in sequence level
            char_embeddeds = torch.cat(char_embeddeds, 1)
            # concatenate word and char-level embedding together in embedding dimension
            embeds = torch.cat([char_embeddeds, embeds], 2)

        packed_embeds = pack_padded_sequence(
            embeds, seq_lens, batch_first=True
        )

        # LSTM output: batch x seq_len x hidden_dim
        lstm_out, self.hidden = self.lstm(packed_embeds, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # generate emission score with linear layer
        lstm_feats = self.hidden2tag(lstm_out)
        # len(sentence) x len(tag_set)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(DEVICE)
        tags = torch.cat(
            [
                torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(
                    DEVICE
                ),
                tags,
            ]
        )
        for i, feat in enumerate(feats):
            tag_vocab.idx2token[tags[i + 1].item()]
            score = (
                score
                + self.transitions[tags[i + 1], tags[i]]
                + feat[tags[i + 1]]
            )
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats, golds=None, cost=None):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.0).to(DEVICE)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for i, feat in enumerate(feats):
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = None
                if cost is not None:
                    # get the cost score
                    cost_score = torch.full(
                        (1, self.tagset_size), cost(golds[i], next_tag)
                    ).to(DEVICE)
                    # add to the score
                    next_tag_var = (
                        forward_var + self.transitions[next_tag] + cost_score
                    )
                else:
                    next_tag_var = forward_var + self.transitions[next_tag]

                assert next_tag_var != None

                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags, seq_lens):
        # loss function: negative log likelihood
        # emission score: seq_len x batch_size x len(tag_set)
        feats_tensor = self._get_lstm_features(sentence, seq_lens)
        loss = torch.tensor(0, dtype=torch.long)
        # go other batch dimension
        for i in range(feats_tensor.size()[0]):
            feats = feats_tensor[i, : seq_lens[i], :]
            tag_seq = tags[i, : seq_lens[i]]
            current_loss = None
            if self.loss_type in "softmax_margin_loss":
                # soft margin loss = - gold score + normalizer(log_sum_exp (score + cost))
                forward_score = self._forward_alg(feats, tag_seq, self.cost)
                gold_score = self._score_sentence(feats, tag_seq)
                current_loss = forward_score - gold_score
            elif self.loss_type == "svm_loss":
                # svm loss = - gold score + max(score + cost)
                viterbi_score, _ = self._viterbi_decode(
                    feats, tag_seq, self.cost
                )
                gold_score = self._score_sentence(feats, tag_seq)
                current_loss = viterbi_score - gold_score
            elif self.loss_type == "ramp_loss":
                # ramp loss = - max(score) + max(score + cost)
                viterbi_score, _ = self._viterbi_decode(feats)
                viterbi_score_with_cost, _ = self._viterbi_decode(
                    feats, tag_seq, self.cost
                )
                current_loss = viterbi_score_with_cost - viterbi_score
            elif self.loss_type == "soft_ramp_loss":
                # soft ramp loss = - log_sum_exp (score) + log_sum_exp (score + cost)
                forward_score = self._forward_alg(feats)
                forward_score_with_cost = self._forward_alg(
                    feats, tag_seq, self.cost
                )
                current_loss = forward_score_with_cost - forward_score
            else:
                # crf loss = - gold score + normalizer(log_sum_exp (score))
                forward_score = self._forward_alg(feats, tag_seq)
                gold_score = self._score_sentence(feats, tag_seq)
                current_loss = forward_score - gold_score

            assert current_loss != None
            loss = loss + current_loss
        return loss

    def forward(
        self, sentence, seq_lens
    ):  # dont confuse this with _forward_alg above.
        scores, preds = [], []
        # Get the "emission scores" from the BiLSTM
        lstm_feats_tensor = self._get_lstm_features(sentence, seq_lens)
        for i in range(lstm_feats_tensor.size()[0]):
            lstm_feats = lstm_feats_tensor[i, : seq_lens[i], :]
            # Find the best path, given the features.
            score, tag_seq = self._viterbi_decode(lstm_feats)
            scores += [score]
            preds += [tag_seq]
        return scores, preds


class CharCNN(nn.Module):
    def __init__(
        self,
        stride=2,
        kernel=2,
        embedding_dim=4,
        max_word_len=20,
    ):
        super(CharCNN, self).__init__()

        # Parameters regarding text preprocessing
        self.embedding_dim = embedding_dim
        self.max_word_len = max_word_len
        self.vocab_size = len(char_vocab.token2idx)

        # Dropout definition
        self.dropout = nn.Dropout(0.25)

        # CNN parameters definition
        self.kernel = kernel
        self.stride = stride
        self.padding = self.kernel - 1

        # Embedding layer definition:
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_dim,
            padding_idx=0,
        )
        # Convolution layer definition
        self.conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
        )

        self.output_dim = (
            self.max_word_len + 2 * self.padding - (self.kernel - 1) - 1
        ) // self.stride + 1
        # Max pooling layers definition
        self.pool = nn.MaxPool1d(self.output_dim, 1)

    def forward(self, X):
        # X: input token
        embedded = self.embedding(X)
        embedded = torch.transpose(embedded, 1, 2)
        embedded = self.dropout(embedded)
        conv_out = self.conv(embedded)
        pool_out = self.pool(conv_out)
        return pool_out