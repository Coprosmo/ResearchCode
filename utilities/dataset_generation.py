from .data_structures import Tree, Node

from collections import deque
import re
import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader


def bfs_visit(tree, store_index=False):
    index = 0
    queue = deque()
    visit_order = []
    queue.append(tree)

    if store_index:
        tree.root.index = index
        index += 1

    while queue:
        x = queue.popleft()
        visit_order.append(x.root)

        for subtree in x.subtrees:
            queue.append(subtree)

            if store_index:
                subtree.root.index = index
                index += 1

    return tuple(visit_order)

#
# def get_subexpressions(tree, store=False):
#     index = 0
#     queue = deque()
#     visit_order = []
#     subexpr = []
#     queue.append(tree)
#
#     if store:
#         tree.root.index = index
#         index += 1
#
#     while queue:
#         x = queue.popleft()
#         visit_order.append(x.root)
#
#         for subtree in x.subtrees:
#             queue.append(subtree)
#
#             if store_index:
#                 subtree.root.index = index
#                 index += 1
#
#     return tuple(visit_order)


def process_theorem(theorem):
    x = theorem
    x = x.replace(r"\'", "'")
    x = x.replace('(', ' ( ')
    x = x.replace(')', ' ) ')
    x = x.split()
    return x


def get_fingerprints(proof_section):
    """ Find fingerprints of goal and all subgoals

    args:
        proof_section - "nodes" object in proof. Contains a subgoal and the tactic used to prove it
    """

    theorem_fingerprint = None
    getting_subgoals = False
    child_fingerprints = []
    split_section = proof_section.split()
    for idx, token in enumerate(split_section):
        if token == 'subgoals':
            getting_subgoals = True
        if token == 'fingerprint:':
            if theorem_fingerprint is None:
                theorem_fingerprint = split_section[idx + 1]
            elif getting_subgoals is True:
                child_fingerprints.append(split_section[idx + 1])
    return theorem_fingerprint, child_fingerprints


def get_theorems(proof):
    """ Make a tree of theorems (string form) and subgoals. Return this tree and all individual theorems.
    """
    tree = None
    theorems = []
    index = dict()
    x = re.split('theorem_in_database', proof)
    proof = x[0]
    m = re.split('nodes', proof)
    for section in m:
        if section:
            # Match exact contents of theorem in form, 'conclusion: " . . . " '
            theorem = re.search('conclusion: "([^"]*)"', section)
            theorems.append(theorem.group(1))
            theorem = theorem.group(1)
            fingerprint, children = get_fingerprints(section)  # fingerprint matches theorem, can be used to index into tree

            if tree is None:
                # Create root node for proof tree, i.e. final conclusion, with subgoals
                tree = Tree(root=Node(
                    label=fingerprint,
                    value=theorem
                ))
                index[fingerprint] = tree.root

                for child in children:
                    subtree = Tree(root=Node(label=child),
                                   parent=tree)
                    tree.add_subtree(subtree)
                    index[child] = subtree

            else:
                # Add subtree with subgoals
                subtree = index[fingerprint]
                subtree.root.value = theorem  # Only fingerprint is stored initially. We now have theorem, so update.

                for child in children:
                    subsubtree = Tree(root=Node(label=child),
                                      parent=subtree)
                    subtree.add_subtree(subsubtree)
                    index[child] = subsubtree

    return theorems, tree


def thm_to_tree(theorem):
    """Transform theorem from string form to tree form. Return tree and a list of the unique distinct values in tree."""

    distinct_features = set()
    tree = Tree(root='', parent=None)
    current_tree = tree
    i_sym = 0
    while i_sym < len(theorem):
        sym = theorem[i_sym]
        if sym == '(':
            new_subtree = Tree(root=Node(theorem[i_sym + 1]), parent=current_tree)
            current_tree.add_subtree(new_subtree)
            current_tree = new_subtree
            # i_sym += 1
        elif sym == ')':
            current_tree.subtree_repr = bfs_visit(current_tree)
            current_tree = current_tree.parents[0]
        else:
            distinct_features.add(sym)
            current_tree.add_subtree(Tree(root=Node(theorem[i_sym]), parent=current_tree))

        i_sym += 1

    final_tree = tree.subtrees[0]
    final_tree.parents = None

    bfs_visit(final_tree, store_index=True)
    return final_tree, distinct_features


def merge_subexpressions(tree):
    subexpressions = dict()
    stack = []
    stack.append(tree)

    while stack:
        t = stack.pop()
        if t.subtree_repr in subexpressions:
            parent = t.parents[0]
            for i, subtree in enumerate(parent.subtrees):
                if subtree.subtree_repr == t.subtree_repr:
                    parent.subtrees[i] = subexpressions[t.subtree_repr]
                    if parent not in subexpressions[t.subtree_repr].parents:
                        subexpressions[t.subtree_repr].parents.append(parent)
        else:
            for subtree in t.subtrees:
                stack.append(subtree)
    return tree


def graph_to_data(tree, distinct_features):
    edges = []
    features = []

    stack = []
    stack.append(tree)
    processed_subtrees = []
    while stack:
        x = stack.pop()
        features.append(x.root.label)

        if x.parents:
            for parent in x.parents:
                edges.append([parent.root.index, x.root.index])

        for subtree in x.subtrees[::-1]:
            if subtree in processed_subtrees:
                continue

            stack.append(subtree)
            edges.append([x.root.index, subtree.root.index])

    features = torch.tensor([[distinct_features.index(x)] for x in features])

    edges = torch.tensor(edges)
    edges = edges.permute(1, 0)

    # datum = Data(x=features, edge_index=edges)

    return features, edges


def make_data():
    datapoints = []
    for i in range(600):
        if i % 10 == 0:
            print(i)
        label = str(i)
        if i // 10 == 0:
            label = '0' + label
        if i // 100 == 0:
            label = '0' + label
        with open(f'../deephol-data/deepmath/deephol/proofs/human/train/prooflogs-00{label}-of-00600.pbtxt', 'r') as f:
            for line in f:
                theorems, tree = get_theorems(line)
#                 size = min(len(tree), 11)
                size = int(len(tree) <= 5)
#                 y = torch.tensor([int(i+1 == size) for i in range(11)]).float()
                datapoints.append((tree.root.value, size))
    return datapoints

