import itertools

from tabulate import tabulate


class Rule:
    """[Rule for the Grammar]
    Rule: left -> right
    """

    def __init__(self, left, right):
        self.left = left
        self.right = right


class Grammar:
    """[Grammar]
    using map for storing rules
    """

    def __init__(self, rules):
        # for top-down parsing
        self.map = {}
        # for bottom-up parsing
        self.reversed_map = {}
        for rule in rules:
            # using list for storing every possible rules given a single key
            self.map[rule.left] = self.map.get(rule.left, set()).union(
                set([rule.right])
            )
            self.reversed_map[rule.right] = self.reversed_map.get(
                rule.right, set()
            ).union(set([rule.left]))


def cky(words, grammar):
    # init tables with n x n empty set: n=number of words
    table = [[""] * i + [set()] * (len(words) - i) for i in range(len(words))]
    # go through each word
    for j in range(len(words)):
        entry_set = set()
        # get the next entry(non-terminals) of terminal itself
        for next_entry in grammar.reversed_map.get(words[j], []):
            entry_set.add(next_entry)
        table[j][j] = entry_set
        # go through every previous position
        for i in reversed(range(j)):
            sub_entry_set = set()

            # go through every possible combination
            # in the interval from i to j
            for k in range(i, j):
                # left
                row_set = table[i][k]
                # right
                col_set = table[k + 1][j]
                # get the cross of 2 set in order to get all combinations of sub-constituents
                cross_set = list(itertools.product(row_set, col_set))

                for left, right in cross_set:
                    sub_constituent = f"{left} {right}"
                    # check if the rule exist
                    if sub_constituent in grammar.reversed_map.keys():
                        sub_entry_set = sub_entry_set.union(
                            grammar.reversed_map[sub_constituent]
                        )
            # put all possible entries in to the table
            table[i][j] = sub_entry_set

    return table


if __name__ == "__main__":
    words = "British left waffles on Falklands".split(" ")
    grammar = Grammar(
        [
            Rule("S", "NP VP"),
            Rule("NP", "JJ NP"),
            Rule("VP", "VP NP"),
            Rule("VP", "VP PP"),
            Rule("PP", "P NP"),
            Rule("NP", "British"),
            Rule("JJ", "British"),
            Rule("NP", "left"),
            Rule("VP", "left"),
            Rule("NP", "waffles"),
            Rule("VP", "waffles"),
            Rule("P", "on"),
            Rule("NP", "Falklands"),
        ]
    )
    print(tabulate(cky(words, grammar), headers=words, showindex="always"))
