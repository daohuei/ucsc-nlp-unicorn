import math

from data import gazetteer


class FeatureVector(object):
    def __init__(self, fdict):
        # mapping for features
        self.fdict = fdict

    def times_plus_equal(self, scalar, v2):
        """
        self += scalar * v2
        :param scalar: Double
        :param v2: FeatureVector
        :return: None
        """
        # update current feature vector given a vector v2
        for key, value in v2.fdict.items():
            self.fdict[key] = scalar * value + self.fdict.get(key, 0)

    def dot_product(self, v2):
        """
        Computes the dot product between self and v2.  It is more efficient for v2 to be the smaller vector (fewer
        non-zero entries).
        :param v2: FeatureVector
        :return: Int
        """
        # dot product with another feature vector
        retval = 0
        for key, value in v2.fdict.items():
            retval += value * self.fdict.get(key, 0)
        return retval

    def square(self):
        retvector = FeatureVector({})
        for key, value in self.fdict.items():
            val_sq = value * value
            retvector.fdict[key] = val_sq
        return retvector

    def square_root(self):
        retvector = FeatureVector({})
        for key, value in self.fdict.items():
            val_sq = math.sqrt(value)
            retvector.fdict[key] = val_sq
        return retvector

    def divide(self, v2):
        retvector = FeatureVector({})
        for key, value in v2.fdict.items():
            if value == 0:
                retvector.fdict[key] = 0
            else:
                retvector.fdict[key] = self.fdict.get(key, 0) / value
        return retvector

    def write_to_file(self, filename):
        """
        Writes the feature vector to a file.
        :param filename: String
        :return: None
        """
        print("Writing to " + filename)
        with open(filename, "w", encoding="utf-8") as f:
            features = [(k, v) for k, v in self.fdict.items()]
            features.sort(key=lambda feature: feature[1], reverse=True)
            for key, value in features:
                f.write("{} {}\n".format(key, value))

    def read_from_file(self, filename):
        """
        Reads a feature vector from a file.
        :param filename: String
        :return: None
        """
        self.fdict = {}
        with open(filename, "r") as f:
            for line in f.readlines():
                txt = line.split()
                self.fdict[txt[0]] = float(txt[1])


class Features(object):
    def __init__(self, inputs, feature_names):
        """
        Creates a Features object
        :param inputs: Dictionary from String to an Array of Strings.
            Created in the make_data_point function.
            inputs['tokens'] = Tokens padded with <START> and <STOP>
            inputs['pos'] = POS tags padded with <START> and <STOP>
            inputs['NP_chunk'] = Tags indicating noun phrase chunks, padded with <START> and <STOP>
            inputs['gold_tags'] = DON'T USE! The gold tags padded with <START> and <STOP>
        :param feature_names: Array of Strings.  The list of features to compute.
        """
        self.feature_names = feature_names
        self.inputs = inputs

    def compute_features(self, cur_tag, pre_tag, i):
        """
        Computes the local features for the current tag, the previous tag, and position i
        :param cur_tag: String.  The current tag.
        :param pre_tag: String.  The previous tag.
        :param i: Int. The position
        :return: FeatureVector
        """
        feats = FeatureVector({})
        cur_word = self.inputs["tokens"][i]
        pos_tag = self.inputs["pos"][i]
        is_last = len(self.inputs["tokens"]) - 1 == i

        # 1 cur word
        # i.e. Wi=France+Ti=I-LOC 1.0
        if "current_word" in self.feature_names:
            key = f"Wi={cur_word}+Ti={cur_tag}"
            add_features(feats, key)

        # 2 prev tag
        # i.e. Ti-1=<START>+Ti=I-LOC 1.0
        if "prev_tag" in self.feature_names:
            key = f"Ti-1={pre_tag}+Ti={cur_tag}"
            add_features(feats, key)

        # 3 lowercased
        # i.e. Oi=france+Ti=I-LOC 1.0
        if "lowercase" in self.feature_names:
            key = f"Oi={cur_word.lower()}+Ti={cur_tag}"
            add_features(feats, key)

        # 4 cur pos
        # i.e. Pi=NNP+Ti=I-LOC 1.0
        if "pos_tag" in self.feature_names:
            key = f"Pi={pos_tag}+Ti={cur_tag}"
            add_features(feats, key)

        # 5 shape of word
        # i.e. Si=Aaaaaa+Ti=I-LOC
        if "word_shape" in self.feature_names:
            word_shape = get_word_shape(cur_word)
            key = f"Si={word_shape}+Ti={cur_tag}"
            add_features(feats, key)

        # 6 (1-4 for prev + for next)
        # i.e. Wi-1=<START>+Ti=I-LOC 1.0
        # i.e. Pi-1=<START>+Ti=I-LOC 1.0
        # i.e. Wi+1=and+Ti=I-LOC 1.0

        if "feats_prev_and_next" in self.feature_names:
            prev_word = self.inputs["tokens"][i - 1]
            prev_pos = self.inputs["pos"][i - 1]
            prev_1 = f"Wi-1={prev_word}+Ti={cur_tag}"
            prev_3 = f"Oi-1={prev_word.lower()}+Ti={cur_tag}"
            prev_4 = f"Pi-1={prev_pos}+Ti={cur_tag}"
            add_features(feats, prev_1)
            add_features(feats, prev_3)
            add_features(feats, prev_4)

            if not is_last:
                next_word = self.inputs["tokens"][i + 1]
                next_pos = self.inputs["pos"][i + 1]
                next_1 = f"Wi+1={next_word}+Ti={cur_tag}"
                next_3 = f"Oi+1={next_word.lower()}+Ti={cur_tag}"
                next_4 = f"Pi+1={next_pos}+Ti={cur_tag}"
                add_features(feats, next_1)
                add_features(feats, next_3)
                add_features(feats, next_4)

        # 7 1,3,4 conjoined with pre_tag
        # i.e. Wi+1=and+Ti-1=<START>+Ti=I-LOC 1.0
        if "feat_conjoined" in self.feature_names:
            conjoined_1 = f"Wi={cur_word}+Ti-1={pre_tag}+Ti={cur_tag}"
            conjoined_3 = f"Oi={cur_word.lower()}+Ti-1={pre_tag}+Ti={cur_tag}"
            conjoined_4 = f"Pi={pos_tag}+Ti-1={pre_tag}+Ti={cur_tag}"
            add_features(feats, conjoined_1)
            add_features(feats, conjoined_3)
            add_features(feats, conjoined_4)

        # 8 k=1,2,3,4 prefix
        # i.e. PREi=Fr+Ti=I-LOC 1.0
        # i.e. PREi=Fra+Ti=I-LOC 1.0
        if "prefix_k" in self.feature_names:
            for k in range(4):
                if k > len(cur_word):
                    break
                prefix = cur_word[: k + 1]
                key = f"PREi={prefix}+Ti={cur_tag}"
                add_features(feats, key)

        # 9 gazetteer
        # i.e. GAZi=True+Ti=I-LOC 1.0
        if "gazetteer" in self.feature_names:
            key = f"GAZi={is_gazetteer(cur_word)}+Ti={cur_tag}"
            add_features(feats, key)

        # 10 is capital
        # i.e. CAPi=True+Ti=I-LOC 1.0
        if "capital" in self.feature_names:
            key = f"CAPi={is_capital(cur_word)}+Ti={cur_tag}"
            add_features(feats, key)

        # 11 position
        # i.e. POSi=1+Ti=I-LOC 1.0
        if "position" in self.feature_names:
            key = f"POSi={i+1}+Ti={cur_tag}"
            add_features(feats, key)

        return feats


def add_features(feats, key):
    feats.times_plus_equal(
        1,
        FeatureVector(
            {key: 1},
        ),
    )


def get_word_shape(word):
    shape = ""
    for c in word:
        shape += get_char_shape(c)
    return shape


def get_char_shape(char):
    encoding = ord(char)
    if encoding >= ord("a") and encoding <= ord("z"):
        return "a"

    if encoding >= ord("A") and encoding <= ord("Z"):
        return "A"

    if encoding >= ord("0") and encoding <= ord("9"):
        return "d"

    return char


def is_gazetteer(word):
    if word in gazetteer:
        return "True"
    return "False"


def is_capital(word):
    if len(word) == 0:
        return "False"
    c = ord(word[0])
    if c >= ord("A") and c <= ord("Z"):
        return "True"
    return "False"


def compute_features(tag_seq, input_length, features):
    """
    Compute f(xi, yi)
    :param tag_seq: [tags] already padded with <START> and <STOP>
    :param input_length: input length including the padding <START> and <STOP>
    :param features: func from token index to FeatureVector
    :return:
    """
    # compute feature given sequence
    feats = FeatureVector({})
    for i in range(1, input_length):
        feats.times_plus_equal(
            1, features.compute_features(tag_seq[i], tag_seq[i - 1], i)
        )
    return feats

    # Examples from class (from slides Jan 15, slide 18):
    # x = will to fight
    # y = NN TO VB
    # features(x,y) =
    #  {"wi=will^yi=NN": 1, // "wi="+current_word+"^yi="+current_tag
    # "yi-1=START^yi=NN": 1,
    # "ti=to+^yi=TO": 1,
    # "yi-1=NN+yi=TO": 1,
    # "xi=fight^yi=VB": 1,
    # "yi-1=TO^yi=VB": 1}

    # x = will to fight
    # y = NN TO VBD
    # features(x,y)=
    # {"wi=will^yi=NN": 1,
    # "yi-1=START^yi=NN": 1,
    # "ti=to+^yi=TO": 1,
    # "yi-1=NN+yi=TO": 1,
    # "xi=fight^yi=VBD": 1,
    # "yi-1=TO^yi=VBD": 1}


feature_1_to_4 = ["current_word", "prev_tag", "lowercase", "pos_tag"]
feature_full = feature_1_to_4 + [
    "word_shape",
    "feats_prev_and_next",
    "feat_conjoined",
    "prefix_k",
    "gazetteer",
    "capital",
    "position",
]