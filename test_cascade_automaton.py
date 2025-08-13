# test_cascade_render.py
import re

from data.cascade_automaton import CascadeSystem, cascade_to_dot, render_cascade


def make_tiny_system():
    """
    Build a deterministic 2-automata cascade and overwrite transitions so tests
    don't rely on RNG. Alphabet = {0,1}. Each automaton has states {0,1}.
    """
    sys = CascadeSystem(num_automata=2, states_per_automaton=2, alphabet_size=2, seed=123)

    # Start states
    for a in sys.automata:
        a.start_state = 0

    # Automaton 0 (independent): merge two labels on the same edge
    # 0 --0,1--> 1 ; 1 --0,1--> 0
    a0 = sys.automata[0]
    a0.dependency_states = None
    a0.transitions.clear()
    a0.transitions[(0, 0)] = 1
    a0.transitions[(0, 1)] = 1
    a0.transitions[(1, 0)] = 0
    a0.transitions[(1, 1)] = 0

    # Automaton 1 (dependent on A0): show composite labels a|d=dep
    # 0 --0|d=0, 1|d=1--> 1 ; others loop to 0
    a1 = sys.automata[1]
    a1.dependency_states = 2
    a1.transitions.clear()
    a1.transitions[(0, 0, 0)] = 1
    a1.transitions[(0, 1, 1)] = 1
    # remaining keys -> 0 (to keep graph total but not create more merged edges)
    for dep in (0, 1):
        for sym in (0, 1):
            if (0, sym, dep) not in a1.transitions:
                a1.transitions[(0, sym, dep)] = 0
    # also define transitions out of state 1 to something simple
    for dep in (0, 1):
        for sym in (0, 1):
            a1.transitions[(1, sym, dep)] = 1

    return sys


def test_structure_clusters_and_nodes():
    sys = make_tiny_system()
    dot = cascade_to_dot(sys, name="Cascade")

    # Clusters present
    assert 'subgraph cluster_0' in dot
    assert 'subgraph cluster_1' in dot

    # Node declarations with labels "i:q"
    assert 'A0_s0 [label="0:0"]' in dot
    assert 'A0_s1 [label="0:1"]' in dot
    assert 'A1_s0 [label="1:0"]' in dot
    assert 'A1_s1 [label="1:1"]' in dot

    # Start arrows per automaton
    assert '"start_0" -> A0_s0;' in dot
    assert '"start_1" -> A1_s0;' in dot


def test_independent_labels_merged():
    sys = make_tiny_system()
    dot = cascade_to_dot(sys, name="Cascade")

    # For automaton 0: 0 -> 1 with merged labels "0,1"
    assert 'A0_s0 -> A0_s1 [label="0,1"]' in dot
    # And 1 -> 0 with merged labels "0,1"
    assert 'A0_s1 -> A0_s0 [label="0,1"]' in dot


def test_dependent_labels_format_and_merge():
    sys = make_tiny_system()
    dot = cascade_to_dot(sys, name="Cascade")

    # Edge A1: 0 -> 1 must contain both "0|d=0" and "1|d=1" in the same label
    m = re.search(r'A1_s0 -> A1_s1 \[label="([^"]+)"\];', dot)
    assert m, "Expected an edge A1_s0 -> A1_s1"
    label = m.group(1)
    assert "0|d=0" in label and "1|d=1" in label


def test_highlighting_with_sequence():
    sys = make_tiny_system()
    # Sequence [0] triggers:
    #  A0: 0 --0--> 1
    #  A1: 0 --0|d=0--> 1  (since dep = current A0 state is 0 at this step)
    dot = cascade_to_dot(sys, name="Cascade", sequence=[0])

    # Filled nodes (visited)
    assert 'A0_s0 [label="0:0", style=filled, fillcolor="lightgray"];' in dot
    assert 'A0_s1 [label="0:1", style=filled, fillcolor="lightgray"];' in dot
    assert 'A1_s0 [label="1:0", style=filled, fillcolor="lightgray"];' in dot
    assert 'A1_s1 [label="1:1", style=filled, fillcolor="lightgray"];' in dot

    # Thick edges (used)
    assert 'A0_s0 -> A0_s1 [label="0,1", penwidth=2];' in dot
    # Dependent edge should be thick as well (label order may vary)
    assert re.search(r'A1_s0 -> A1_s1 \[label="[^"]+", penwidth=2\];', dot)