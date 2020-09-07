from .data_structures import Tree, Node

from collections import deque
import re
import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from tqdm import tqdm
import random

random.seed(42)


def bfs_visit(tree, store_index=False, fix_subtrees=True):
    index = 0
    queue = deque()
    visit_order = []
    queue.append(tree)
    visited = [tree]

    if store_index:
        tree.root.index = index
        index += 1

    while queue:
        x = queue.popleft()
        visit_order.append(x.root)
        
        if fix_subtrees and x.subtree_str is None:
            x.subtree_str = x.root.label

        for subtree in x.subtrees:
            if subtree not in visited:
                queue.append(subtree)
                visited.append(subtree)

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


def thm_to_tree(theorem, to_merge):
    """Transform theorem from string form to tree form. Return tree and a list of the unique distinct values in tree."""

    distinct_features = set(x for x in theorem if x not in '()')
#     print(theorem, distinct_features)
    tree = Tree(root='', parent=None)
    current_tree = tree
    i_sym = 0
    while i_sym < len(theorem):
        sym = theorem[i_sym]

        if sym == '(':
            new_subtree = Tree(root=Node(theorem[i_sym + 1]), parent=current_tree, thm_start_idx=(i_sym + 1))
            current_tree.add_subtree(new_subtree)
            current_tree = new_subtree
            i_sym += 1
        elif sym == ')':
            current_tree.subtree_repr = bfs_visit(current_tree)
            current_tree.subtree_str = ' '.join(theorem[current_tree.thm_start_idx : i_sym])
            current_tree = current_tree.parents[0]
        else:
            current_tree.add_subtree(Tree(root=Node(theorem[i_sym]), parent=current_tree))

        i_sym += 1

    final_tree = tree.subtrees[0]
    final_tree.parents = None
    
    if to_merge:
        bfs_visit(final_tree, store_index=False, fix_subtrees=True)
    else:
        bfs_visit(final_tree, store_index=True, fix_subtrees=True)
        
    return final_tree, distinct_features


def merge_subexpressions(tree):
    """Merge all similar subtrees within a tree. 
    
    Similar subtrees are identified by a DFS traversal string. The process
    is as follows: when a subtree T with parent p is found to be similar to 
    a subtree T' with parent p', all of ps subtrees which are similar to T'
    are set to T', and p is added to the list of parents of T'.
    """
    
    subexpressions = dict()
    stack = []
    stack.append((tree, None))

    while stack:
        t, child_index = stack.pop()

        if t.subtree_str in subexpressions:
            parent = t.parents[0]
            parent.subtrees[child_index] = subexpressions[t.subtree_str]
            subexpressions[t.subtree_str].parents.append(parent)

        else:
            subexpressions[t.subtree_str] = t
            for idx, subtree in enumerate(t.subtrees[::-1]):
                stack.append((subtree, (len(t.subtrees)-1)-idx))

    bfs_visit(tree, store_index=True, fix_subtrees=False)
    return tree


def graph_to_data(tree, normalized_features=None):
    edges_up = []
    edges_down = []
    edge_features_up = []
    edge_features_down = []
    node_features = []

    stack = []
    stack.append((tree, 0))
    processed_subtrees = []
    
    while stack:
        x, child_index = stack.pop()
        node_features.append(x.root.label)
        if child_index > 1:
            print('Found node with > 1 children')

        if x.parents:
            for parent in x.parents:
#                 for p, c in edges_up:
#                     if p == parent.root.index and c == x.root.index:
#                         break
#                 else:
                edges_up.append([x.root.index, parent.root.index])
                edge_features_up.append([child_index])

        for idx, subtree in enumerate(x.subtrees[::-1]):
            edges_down.append([x.root.index, subtree.root.index])
            edge_features_down.append([(len(x.subtrees) - 1) - idx])
            
            if subtree not in processed_subtrees:
                stack.append((subtree, (len(x.subtrees) - 1) - idx))
                processed_subtrees.append(subtree)

     
    if normalized_features is not None:
        node_features = torch.tensor([normalized_features[x] for x in node_features])
    
    edges_up = torch.tensor(edges_up)
    edges_up = edges_up.permute(1, 0)
    edges_down = torch.tensor(edges_down)
    edges_down = edges_down.permute(1, 0)
    
    edge_features_up = torch.tensor(edge_features_up)
    edge_features_down = torch.tensor(edge_features_down)

    return node_features, (edges_up, edges_down), (edge_features_up, edge_features_down)


def make_data(binary=False, only_top=True):
    datapoints = []
#     for i in tqdm(range(2)):
    for i in tqdm(range(50)):
#         if i % 15 != 0:
#             continue
        label = str(i)
        if i // 10 == 0:
            label = '0' + label
        if i // 100 == 0:
            label = '0' + label
        with open(f'../deephol-data/deepmath/deephol/proofs/human/train/prooflogs-00{label}-of-00600.pbtxt', 'r') as f:
            for line in f:
                theorems, tree = get_theorems(line)
                
                if only_top:
                    if binary:
                        size = int(len(tree) <= 5)
                    else:
                        size = min(len(tree), 11) - 1
                    datapoints.append((tree.root.value, float(size)))
                    
                else:
                    stack = [tree]
                    while stack:
                        t = stack.pop()
                        if t.root.value is None:
                            continue
                        if 'hypo' in t.root.value:
                            print('Hypothesis Error!!!!')
                        for s in t.subtrees:
                            assert len(t) > len(s)
                            stack.append(s)
                        
                        if binary:
                            t_size = int(len(t) <= 5)
                        else:
                            t_size = min(len(t), 11) - 1
                        datapoints.append((t.root.value, float(t_size)))

    return datapoints


def get_data_from_file(i, binary=False, only_top=True):
    """A modified version of make_data function, which gets all data
    from a specific file.
    """
    
    datapoints = []
    
    label = str(i)
    if i // 10 == 0:
        label = '0' + label
    if i // 100 == 0:
        label = '0' + label
        
    with open(f'../deephol-data/deepmath/deephol/proofs/human/train/prooflogs-00{label}-of-00600.pbtxt', 'r') as f:
        for line in f:
            theorems, tree = get_theorems(line)

            if only_top:
                if binary:
                    size = int(len(tree) <= 5)
                else:
                    size = min(len(tree), 11) - 1
                datapoints.append((tree.root.value, float(size)))

            else:
                stack = [tree]
                while stack:
                    t = stack.pop()
                    if t.root.value is None:
                        continue
                    if 'hypo' in t.root.value:
                        print('Hypothesis Error!!!!')
                    for s in t.subtrees:
                        assert len(t) > len(s)
                        stack.append(s)

                    if binary:
                        t_size = int(len(t) <= 5)
                    else:
                        t_size = min(len(t), 11) - 1
                    datapoints.append((t.root.value, float(t_size)))

    return datapoints
