from random import sample


def read_data(filename):
    """
    Reads the CoNLL 2003 data into an array of dictionaries (a dictionary for each data point).
    :param filename: String
    :return: Array of dictionaries.  Each dictionary has the format returned by the make_data_point function.
    """
    data = []
    with open(filename, "r") as f:
        sent = []
        for line in f.readlines():
            if line.strip():
                sent.append(line)
            else:
                data.append(make_data_point(sent))
                sent = []
        data.append(make_data_point(sent))

    return data


def make_data_point(sent):
    """
        Creates a dictionary from String to an Array of Strings representing the data.  The dictionary items are:
        dic['tokens'] = Tokens padded with <START> and <STOP>
        dic['pos'] = POS tags padded with <START> and <STOP>
        dic['NP_chunk'] = Tags indicating noun phrase chunks, padded with <START> and <STOP> (but will not use)
        dic['gold_tags'] = The gold tags padded with <START> and <STOP>
    :param sent: String.  The input CoNLL format string
    :return: Dict from String to Array of Strings.
    """
    dic = {}
    sent = [s.strip().split() for s in sent]
    dic["tokens"] = ["<START>"] + [s[0] for s in sent] + ["<STOP>"]
    dic["pos"] = ["<START>"] + [s[1] for s in sent] + ["<STOP>"]
    dic["NP_chunk"] = ["<START>"] + [s[2] for s in sent] + ["<STOP>"]
    dic["gold_tags"] = ["<START>"] + [s[3] for s in sent] + ["<STOP>"]
    return dic


def read_gazetteer():
    data = []
    with open("gazetteer.txt", "r") as f:
        for line in f.readlines():
            data += line.split()[1:]
    return data


gazetteer = read_gazetteer()
sample_num = 500
train_data = read_data("ner.train")
dev_data = sample(read_data("ner.dev"), 150)
test_data = sample(read_data("ner.test"), 150)
tagset = [
    "B-PER",
    "B-LOC",
    "B-ORG",
    "B-MISC",
    "I-PER",
    "I-LOC",
    "I-ORG",
    "I-MISC",
    "O",
]