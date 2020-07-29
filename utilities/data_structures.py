class Tree:
    """Used to represent theorems graphically. Tree recursively defined as a root with possibly several child
    subtrees. Multiple parents are supported, enabling subexpression sharing. Tree can also be used to represent a
    proof tree (where nodes are theorems and children are subgoals), though currently this is not actively used.

    args:
        root - object of type Node
        parents - list of Tree objects
        subtrees - list of Tree objects
        subtree_repr - string representation of tree
        thm_start_idx - position in theorem of the start of the tree
    """

    def __init__(self, root, parent=None, thm_start_idx=0):
        self.root = root
        self.parents = [parent] if parent else []
        self.thm_start_idx = thm_start_idx
        self.subtrees = []
        self.subtree_repr = None

    def add_subtrees(self, *subtrees):
        self.subtrees.extend(subtrees)

    def add_subtree(self, subtree):
        self.subtrees.append(subtree)

    def view_subtree(self):
        return self.subtree_repr

    def __len__(self):
        n_nodes = 0
        q = []
        q.append(self)
        seen = [self]
        while q:
            node = q.pop(0)
            n_nodes += 1
            for child in node.subtrees:
                if child in seen:
                    continue
                else:
                    q.append(child)
                    seen.append(child)
        return n_nodes

    def __str__(self):
        return f'Tree(root={self.root}, parents={self.parents}, size={self.__len__()})'


class Node:
    """Node object used to store a symbol in a theorem. Intended for use with Tree object.

    args:
        label - unique identifier for node
        index - unique position in tree
        value - node data, not necessarily unique
        subtree_str - string representation of subtree, determined via dfs traversal
    """

    def __init__(self, label, index=None, value=None, subtree_str=None):
        self.label = label
        self.index = index
        self.value = value
        self.subtree_str = subtree_str

    def __str__(self):
        return f'({self.label}, index={self.index})'

    def __repr__(self):
        return f'({self.label}, index={self.index})'