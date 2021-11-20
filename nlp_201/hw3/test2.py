from taggers import HMMTagger, BaselineTagger
from starter_code import evaluate

train_sample = "The/DT House/NNP joined/VBD  the/DT Senate/NNP in/IN making/VBG  federal/JJ reparations/NNS for/IN Japanese-Americans/NNPS"
test_samples = [
    "alsjfla the askdmc and djdsaas in making",
    "vsacs the House and djdsaas in giuhun",
    "The House joined  the Senate in making  federal reparations for Japanese-Americans",
]
train_corpus = [train_sample]
tagger = HMMTagger()
tagger.train(train_corpus)
hmm_y = tagger.predict(test_samples)

b_tagger = BaselineTagger()
b_tagger.train(train_corpus)
b_y = b_tagger.predict(test_samples)


def remove_start_stop_tokens(sent):
    return sent[1:-1]


print(
    evaluate(
        [remove_start_stop_tokens(sent) for sent in hmm_y],
        [remove_start_stop_tokens(sent) for sent in b_y],
    )
)
