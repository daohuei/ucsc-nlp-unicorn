import itertools
import math

from tabulate import tabulate
import numpy as np

from print_tree import print_tree


class ProbRule:
    """[Rule for the Grammar]
    Rule: left -> right
    """

    def __init__(self, left, right, prob):
        self.left = left
        self.right = right
        self.prob = prob


class ProbGrammar:
    """[Grammar]
    using map for storing rules
    """

    def __init__(self, rules):
        # for bottom-up parsing
        self.reversed_map = {}
        for rule in rules:
            # using list for storing every possible rules given a single key
            self.reversed_map[rule.right] = self.reversed_map.get(
                rule.right, set()
            ).union(set([(rule.left, rule.prob)]))


class SyntaxTreeNode:
    def __init__(self, non_terminal, log_prob, left, right, word=None):
        # if it is the leaf of the tree, let it attach to the word string
        self.value = non_terminal if word is None else f"{non_terminal}-{word}"
        self.left, self.right = left, right

        # zero if no leaves
        left_score = 0 if self.left is None else self.left.score
        right_score = 0 if self.right is None else self.right.score

        # calculate the score in the log space
        self.score = log_prob + left_score + right_score
        # calculate back the probability
        self.prob = math.exp(self.score)

    def __repr__(self):
        return str(self.prob)


def cky(words, grammar):
    # init tables with n x n empty set: n=number of words
    table = [[""] * i + [set()] * (len(words) - i) for i in range(len(words))]
    trees = []
    # go through each word
    for j in range(len(words)):
        entry_set = set()
        # get the next entry(non-terminals) of terminal itself
        for next_entry, prob in grammar.reversed_map.get(words[j], []):
            entry_node = SyntaxTreeNode(
                next_entry, np.log(prob), None, None, word=words[j]
            )
            entry_set.add((next_entry, entry_node))
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

                for left_tuple, right_tuple in cross_set:
                    # get the left node and the right node
                    left, left_node = left_tuple
                    right, right_node = right_tuple
                    # concat them as the sub-constituent
                    sub_constituent = f"{left} {right}"
                    # check if the rule exist
                    if sub_constituent in grammar.reversed_map.keys():
                        # get the entry non-terminal and the probability pair from Grammar hashmap
                        for sub_entry, sub_prob in grammar.reversed_map[
                            sub_constituent
                        ]:
                            #
                            sub_entry_node = SyntaxTreeNode(
                                sub_entry,
                                np.log(sub_prob),
                                left_node,
                                right_node,
                            )
                            sub_entry_set.add(
                                (
                                    sub_entry,
                                    sub_entry_node,
                                )
                            )
            # put all possible entries in to the table
            table[i][j] = sub_entry_set
    for tag, root in table[0][-1]:
        # full syntax tree
        if tag == "S":
            trees.append(root)

    return table, trees


def marginalize_trees(trees):
    if not trees:
        return 0
    prob = sum([tree.prob for tree in trees])
    return prob


if __name__ == "__main__":
    words = "astronomers saw stars with ears".split(" ")
    grammar = ProbGrammar(
        [
            ProbRule("S", "NP VP", 1.0),
            ProbRule("PP", "P NP", 1.0),
            ProbRule("VP", "V NP", 0.7),
            ProbRule("VP", "VP PP", 0.3),
            ProbRule("P", "with", 1.0),
            ProbRule("V", "saw", 1.0),
            ProbRule("NP", "NP PP", 0.4),
            ProbRule("NP", "astronomers", 0.4),
            ProbRule("NP", "ears", 0.18),
            ProbRule("NP", "saw", 0.04),
            ProbRule("NP", "stars", 0.18),
            ProbRule("NP", "telescopes", 0.1),
        ]
    )
    # parse the sentences with grammar with CKY algorithm and return the tables and syntax trees
    tables, trees = cky(words, grammar)
    best_tree = max(trees, key=lambda node: node.score)
    print(tabulate(tables, headers=words, showindex="always"))
    # pretty print the best tree
    print_tree(best_tree)
    # marginalize over all results trees
    print(marginalize_trees(trees))
