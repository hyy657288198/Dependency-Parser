#!/usr/bin/env python3
# Student name: Yiyang Hua
# Student number: 1003201475
# UTORid: huayiyan
"""Functions and classes that handle parsing"""

from itertools import chain

from nltk.parse import DependencyGraph


class PartialParse(object):
    """A PartialParse is a snapshot of an arc-standard dependency parse

    It is fully defined by a quadruple (sentence, stack, next, arcs).

    sentence is a tuple of ordered pairs of (word, tag), where word
    is a a word string and tag is its part-of-speech tag.

    Index 0 of sentence refers to the special "root" node
    (None, self.root_tag). Index 1 of sentence refers to the sentence's
    first word, index 2 to the second, etc.

    stack is a list of indices referring to elements of
    sentence. The 0-th index of stack should be the bottom of the stack,
    the (-1)-th index is the top of the stack (the side to pop from).

    next is the next index that can be shifted from the buffer to the
    stack. When next == len(sentence), the buffer is empty.

    arcs is a list of triples (idx_head, idx_dep, deprel) signifying the
    dependency relation `idx_head ->_deprel idx_dep`, where idx_head is
    the index of the head word, idx_dep is the index of the dependant,
    and deprel is a string representing the dependency relation label.
    """

    left_arc_id = 0
    """An identifier signifying a left arc transition"""

    right_arc_id = 1
    """An identifier signifying a right arc transition"""

    shift_id = 2
    """An identifier signifying a shift transition"""

    root_tag = "TOP"
    """A POS-tag given exclusively to the root"""

    def __init__(self, sentence):
        # the initial PartialParse of the arc-standard parse
        # **DO NOT ADD ANY MORE ATTRIBUTES TO THIS OBJECT**
        self.sentence = ((None, self.root_tag),) + tuple(sentence)
        self.stack = [0]
        self.next = 1
        self.arcs = []

    @property
    def complete(self):
        """bool: return true iff the PartialParse is complete

        Assume that the PartialParse is valid
        """
        ##****BEGIN YOUR CODE****
        if len(self.stack) == 1 and self.next == len(self.sentence):
            return True
        else:
            return False
        ##****END YOUR CODE****

    def parse_step(self, transition_id, deprel=None):
        """Update the PartialParse with a transition

        Args:
            transition_id : int
                One of left_arc_id, right_arc_id, or shift_id. You
                should check against `self.left_arc_id`,
                `self.right_arc_id`, and `self.shift_id` rather than
                against the values 0, 1, and 2 directly.
            deprel : str or None
                The dependency label to assign to an arc transition
                (either a left-arc or right-arc). Ignored if
                transition_id == shift_id

        Raises:
            ValueError if transition_id is an invalid id or is illegal
                given the current state
        """
        ##****BEGIN YOUR CODE****
        if len(self.stack) == 1 and transition_id != self.shift_id:
            raise ValueError('invalid')
        elif transition_id == self.shift_id and self.next == len(self.sentence):
            raise ValueError('invalid')
        else:
            if transition_id == self.shift_id:
                self.stack += [self.next]
                self.next += 1
            elif transition_id == self.left_arc_id:
                self.arcs += [(self.stack[-1], self.stack[-2], deprel)]
                self.stack.pop(-2)
            elif transition_id == self.right_arc_id:
                self.arcs.append((self.stack[-2], self.stack[-1], deprel))
                self.stack.pop(-1)
            else:
                raise ValueError('invalid')
        ##****END YOUR CODE****

    def get_n_leftmost_deps(self, sentence_idx, n=None):
        """Returns a list of n leftmost dependants of word

        Leftmost means closest to the beginning of the sentence.

        Note that only the direct dependants of the word on the stack
        are returned (i.e. no dependants of dependants).

        Args:
            sentence_idx : refers to word at self.sentence[sentence_idx]
            n : the number of dependants to return. "None" refers to all
                dependants

        Returns:
            deps : The n leftmost dependants as sentence indices.
                If fewer than n, return all dependants. Return in order
                with the leftmost @ 0, immediately right of leftmost @
                1, etc.
        """
        ##****BEGIN YOUR CODE****
        result = []
        for i in self.arcs:
            if i[0] == sentence_idx:
                result += [i[1]]
        if n is None or n > len(result):
            deps = result
        else:
            deps = result[:n]
        ##****END YOUR CODE****
        return deps

    def get_n_rightmost_deps(self, sentence_idx, n=None):
        """Returns a list of n rightmost dependants of word on the stack @ idx

        Rightmost means closest to the end of the sentence.

        Note that only the direct dependants of the word on the stack
        are returned (i.e. no dependants of dependants).

        Args:
            sentence_idx : refers to word at self.sentence[sentence_idx]
            n : the number of dependants to return. "None" refers to all
                dependants

        Returns:
            deps : The n rightmost dependants as sentence indices. If
                fewer than n, return all dependants. Return in order
                with the rightmost @ 0, immediately left of rightmost @
                1, etc.
        """
        ##****BEGIN YOUR CODE****
        result = []
        i = len(self.arcs)-1
        while i > -1:
            if self.arcs[i][0] == sentence_idx:
                result += [self.arcs[i][1]]
            i -= 1
        if n is None or n > len(result):
            deps = result
        else:
            deps = result[:n]
        ##****END YOUR CODE****
        return deps

    def get_oracle(self, graph: DependencyGraph):
        """Given a projective dependency graph, determine an appropriate
        transition

        This method chooses either a left-arc, right-arc, or shift so
        that, after repeated calls to pp.parse_step(*pp.get_oracle(graph)),
        the arc-transitions this object models matches the
        DependencyGraph "graph". For arcs, it also has to pick out the
        correct dependency relationship.
        graph is projective: informally, this means no crossed lines in the
        dependency graph. More formally, if i -> j and j -> k, then:
             if i > j (left-arc), i > k
             if i < j (right-arc), i < k

        You don't need to worry about API specifics about graph; just call the
        relevant helper functions from the HELPER FUNCTIONS section below. In
        particular, you will (probably) need:
         - get_deprel(i, graph), which will return the dependency relation
           label for the word at index i
         - get_head(i, graph), which will return the index of the head word for
           the word at index i
         - get_deps(i, graph), which will return the indices of the dependants
           of the word at index i

        Hint: take a look at get_left_deps and get_right_deps below; their
        implementations may help or give you ideas even if you don't need to
        call the functions themselves.

        *IMPORTANT* if left-arc and shift operations are both valid and
        can lead to the same graph, always choose the left-arc
        operation.

        *ALSO IMPORTANT* make sure to use the values `self.left_arc_id`,
        `self.right_arc_id`, `self.shift_id` for the transition rather than
        0, 1, and 2 directly

        Args:
            graph : nltk.parse.dependencygraph.DependencyGraph
                A projective dependency graph to head towards

        Returns:
            transition, deprel_label : the next transition to take, along
                with the correct dependency relation label; if transition
                indicates shift, deprel_label should be None

        Raises:
            ValueError if already completed. Otherwise you can always
            assume that a valid move exists that heads towards the
            target graph
        """
        if self.complete:
            raise ValueError('PartialParse already completed')
        transition, deprel_label = -1, None
        ##****BEGIN YOUR CODE****
        self.sentence = get_sentence_from_graph(graph, True)
        last = self.stack[-1]

        left_list = []
        for dep in get_left_deps(last, graph):
            left_list += [dep]

        right_list = []
        for dep in get_right_deps(last, graph):
            right_list += [dep]

        max_left = None
        while len(left_list) != 0:
            if max(left_list) in self.stack:
                max_left = max(left_list)
                break
            left_list.remove(max(left_list))

        min_right = None
        while len(right_list) != 0:
            if min(right_list) >= self.next:
                min_right = min(right_list)
                break
            right_list.remove(min(right_list))

        if graph.nodes[last]['head'] is None:
            graph.nodes[last]['head'] = 0

        if last in self.stack:
            # left-arc
            if max_left and len(self.stack) > 2:
                temp = graph.nodes[last]['deps']
                for deprel_label in temp:
                    if max_left in temp[deprel_label]:
                        transition = self.left_arc_id
                        deprel_label = deprel_label
                        break
            # right-arc
            elif min_right is None and graph.nodes[last]['head'] in self.stack and len(self.stack) > 1:
                temp = graph.nodes[graph.nodes[last]['head']]['deps']
                for deprel_label in temp:
                    if last in temp[deprel_label]:
                        transition = self.right_arc_id
                        deprel_label = deprel_label
                        break
            # shift
            elif len(self.sentence) != self.next:
                transition = self.shift_id

        ##****END YOUR CODE****
        return transition, deprel_label

    def parse(self, td_pairs):
        """Applies the provided transitions/deprels to this PartialParse

        Simply reapplies parse_step for every element in td_pairs

        Args:
            td_pairs:
                The list of (transition_id, deprel) pairs in the order
                they should be applied
        Returns:
            The list of arcs produced when parsing the sentence.
            Represented as a list of tuples where each tuple is of
            the form (head, dependent)
        """
        for transition_id, deprel in td_pairs:
            self.parse_step(transition_id, deprel)
        return self.arcs


def minibatch_parse(sentences, model, batch_size):
    """Parses a list of sentences in minibatches using a model.

    Note that parse_step may raise a ValueError if your model predicts
    an illegal (transition, label) pair. Remove any such `stuck`
    partial-parses from the list unfinished_parses.

    Args:
        sentences:
            A list of "sentences", where each element is itself a list
            of pairs of (word, pos)
        model:
            The model that makes parsing decisions. It is assumed to
            have a function model.predict(partial_parses) that takes in
            a list of PartialParse as input and returns a list of
            pairs of (transition_id, deprel) predicted for each parse.
            That is, after calling
                td_pairs = model.predict(partial_parses)
            td_pairs[i] will be the next transition/deprel pair to apply
            to partial_parses[i].
        batch_size:
            The number of PartialParse to include in each minibatch
    Returns:
        arcs:
            A list where each element is the arcs list for a parsed
            sentence. Ordering should be the same as in sentences (i.e.,
            arcs[i] should contain the arcs for sentences[i]).
    """
    ##****BEGIN YOUR CODE****
    result = []
    temp = []
    for word in sentences:
        parse = PartialParse(word)
        result += [parse]
        temp += [parse]

    while temp:
        minibatch = temp[:batch_size]
        transitions = model.predict(minibatch)

        for i, j in enumerate(transitions):
            try:
                minibatch[i].parse_step(j[0], j[1])
                if minibatch[i].complete:
                    temp.remove(minibatch[i])
            except ValueError:
                temp.remove(minibatch[i])

    arcs = []
    for word in result:
        arcs += [word.arcs]

    ##****END YOUR CODE****
    return arcs


# ****HELPER FUNCTIONS (look here!)****


def get_deprel(sentence_idx: int, graph: DependencyGraph):
    """Get the dependency relation label for the word at index sentence_idx
    from the provided DependencyGraph"""
    return graph.nodes[sentence_idx]['rel']


def get_head(sentence_idx: int, graph: DependencyGraph):
    """Get the index of the head of the word at index sentence_idx from the
    provided DependencyGraph"""
    return graph.nodes[sentence_idx]['head']


def get_deps(sentence_idx: int, graph: DependencyGraph):
    """Get the indices of the dependants of the word at index sentence_idx
    from the provided DependencyGraph"""
    return list(chain(*graph.nodes[sentence_idx]['deps'].values()))


def get_left_deps(sentence_idx: int, graph: DependencyGraph):
    """Get the arc-left dependants of the word at index sentence_idx from
    the provided DependencyGraph"""
    return (dep for dep in get_deps(sentence_idx, graph)
            if dep < graph.nodes[sentence_idx]['address'])


def get_right_deps(sentence_idx: int, graph: DependencyGraph):
    """Get the arc-right dependants of the word at index sentence_idx from
    the provided DependencyGraph"""
    return (dep for dep in get_deps(sentence_idx, graph)
            if dep > graph.nodes[sentence_idx]['address'])


def get_sentence_from_graph(graph, include_root=False):
    """Get the associated sentence from a DependencyGraph"""
    sentence_w_addresses = [(node['address'], node['word'], node['ctag'])
                            for node in graph.nodes.values()
                            if include_root or node['word'] is not None]
    sentence_w_addresses.sort()
    return tuple(t[1:] for t in sentence_w_addresses)


##****TESTING****

def _test_arcs(name, pp, ex_arcs):
    """Tests the provided arcs match the expected arcs"""
    arcs = tuple(sorted(pp.arcs))
    ex_arcs = tuple(sorted(ex_arcs))
    assert arcs == ex_arcs, \
        "{} test resulted in arc list {}, expected {}".format(
            name,
            [(pp.sentence[arc[0]], pp.sentence[arc[1]], arc[2])
             for arc in arcs],
            [(pp.sentence[arc[0]], pp.sentence[arc[1]], arc[2])
             for arc in ex_arcs]
            )


def test_oracle_q2d():
    """(2d) Custom test case for oracle.

    Fill in your custom test case in this function.
    You will likely find it easiest to just edit graph_data, the argument to
    _test_arcs, and ex_tids.
    """
    ##****BEGIN YOUR CODE****
    graph_data = """\
A DET 3 det
record NOUN 3 compound
date NOUN 7 nsubjpass
has AUX 7 aux
n't PART 7 neg
been AUX 7 auxpass
set VERB 0 ROOT
. PUNCT 7 punct
"""
    graph = DependencyGraph(graph_data)
    pp = PartialParse(get_sentence_from_graph(graph))
    transition_ids = []
    while not pp.complete:
        transition_id, deprel = pp.get_oracle(graph)
        transition_ids.append(transition_id)
        pp.parse_step(transition_id, deprel)
    _test_arcs("oracle", pp,
               [(3, 1, 'det'), (3, 2, 'compound'), (7, 3, 'nsubjpass'),
                (7, 4, 'aux'), (7, 5, 'neg'), (7, 6, 'auxpass'), (0, 7, 'ROOT'), (7, 8, 'punct')]
               )
    ex_tids = [pp.shift_id, pp.shift_id, pp.shift_id,
               pp.left_arc_id, pp.left_arc_id, pp.shift_id,
               pp.shift_id, pp.shift_id, pp.shift_id,
               pp.left_arc_id, pp.left_arc_id, pp.left_arc_id,
               pp.left_arc_id, pp.shift_id, pp.right_arc_id, pp.right_arc_id
               ]
    assert transition_ids == ex_tids, \
        "oracle test resulted in transitions {}, expected {}".format(
            transition_ids, ex_tids)
    ##****END YOUR CODE****
    print('custom oracle test passed!')


if __name__ == '__main__':
    test_oracle_q2d()
