"""
RNN-to-DFA extraction (fixpoint over A* via ε-clustering on hidden states).

This version:
- NO FiniteLanguage needed.
- Explores A* breadth-first, starting from ε, and adds a new representative
  iff its hidden state is > eps away from all existing reps (your "case 3").
- Stops when the BFS queue empties (i.e., closure: every rep has all A-edges mapped).
- Relabels representatives to 0..N-1 and constructs a FiniteAutomaton.

Author: ChatGPT (adapted per user's request)
"""

from __future__ import annotations
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple
from collections import deque

import sys
import os

import random
import numpy as np
import torch
import torch.nn as nn

# Make local packages importable if this file sits in /extractors/ or similar
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from models.automaton_rnn import AutomatonRNN
from data.automaton import FiniteAutomaton


# =====================
# Utility / Converters
# =====================

def random_words(A: Sequence[Any], n: int, k: int) -> List[str]:
    """
    Generate k random words over alphabet A, each of length n, returned as strings.
    (Used only in the demo.)
    """
    if not A:
        raise ValueError("Alphabet A must be non-empty.")
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative integers.")
    words = [random.choices(list(A), k=n) for _ in range(k)]
    return ["".join(map(str, w)) for w in words]


@torch.no_grad()
def _symbols_to_scalar_sequence(
    symbols: Sequence[int],
    alphabet_size: int
) -> torch.Tensor:
    """
    Convert a list of symbol indices to a float tensor of shape (1, T, 1),
    scaling indices into [0,1] by denom = max(1, alphabet_size-1).
    """
    T = len(symbols)
    if T == 0:
        return torch.empty((1, 0, 1), dtype=torch.float32)
    denom = max(1, alphabet_size - 1)
    x = torch.tensor([[[s / denom] for s in symbols]], dtype=torch.float32)  # (1, T, 1)
    return x


@torch.no_grad()
def get_hidden_state_automaton_rnn(
    model: AutomatonRNN,
    word: Sequence[Any],
    symbol_to_idx: Dict[Any, int],
    alphabet_size: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Return the hidden state h_T after reading 'word' (sequence of symbols).
    Handles the empty word by returning zeros of size model.hidden_size.
    """
    if device is None:
        device = next(model.parameters()).device

    ints = [symbol_to_idx[sym] for sym in word]

    if len(ints) == 0:
        return torch.zeros(model.hidden_size, device=device)

    x = _symbols_to_scalar_sequence(ints, alphabet_size).to(device)  # (1, T, 1)
    model.eval()
    h0 = torch.zeros(1, 1, model.hidden_size, device=device)  # (num_layers=1, B=1, H)
    rnn_out, _ = model.rnn(x, h0)  # (1, T, H)
    h_T = rnn_out[0, -1, :].clone()
    return h_T


# ==================================
# Hidden-state DFA "extraction" core
# ==================================

def build_dfa_until_fixpoint(
    model: AutomatonRNN,
    symbol_to_idx: Dict[Hashable, int],
    alphabet_size: int,
    eps: float,
    *,
    alphabet_symbols: Optional[Iterable[Hashable]] = None,
    max_states: Optional[int] = None,   # safety cap to avoid runaway growth
) -> FiniteAutomaton:
    """
    Explore A* breadth-first, adding a new representative state for a word w
    iff ||h(w) - h(rep)|| >= eps for all existing representatives rep.
    Transitions are recorded between representatives. When the queue empties,
    δ: Reps × A -> Reps is total, so no longer word can trigger a new case-3.

    Args
    ----
    model : AutomatonRNN
        Trained RNN with input_size=1 and attributes: hidden_size, rnn.
    symbol_to_idx : dict
        Map from symbol to integer index used by the encoder.
    alphabet_size : int
        Size of the alphabet (used for scalar encoding).
    eps : float
        ε-threshold for merging hidden states (Euclidean norm).
    alphabet_symbols : iterable (optional)
        The concrete list of symbols to expand on. Defaults to sorted keys of symbol_to_idx.
    max_states : int (optional)
        Hard cap on the number of representatives; if reached, stop.

    Returns
    -------
    FiniteAutomaton
        DFA over the discovered representatives, relabeled to 0..N-1.
    """
    # ----- prepare alphabet (deterministic order helps reproducibility)
    if alphabet_symbols is None:
        alphabet: List[Hashable] = sorted(symbol_to_idx.keys(), key=lambda x: str(x))
    else:
        alphabet = list(alphabet_symbols)

    # ----- caches and structures
    @torch.no_grad()
    def hidden_of(word: str) -> torch.Tensor:
        if word in hidden_cache:
            return hidden_cache[word]
        h = get_hidden_state_automaton_rnn(
            model=model,
            word=list(word),  # each symbol is a character from `alphabet`
            symbol_to_idx=symbol_to_idx,
            alphabet_size=alphabet_size,
        )
        hidden_cache[word] = h
        return h

    def find_rep(h: torch.Tensor) -> Optional[str]:
        """
        Return an existing representative whose hidden state is within eps of h,
        else None. (Greedy: pick the closest within eps if multiple.)
        """
        best: Optional[str] = None
        best_dist = float("inf")
        for s_rep, (h_rep, _cls) in D.items():
            d = torch.norm(h - h_rep).item()
            if d < eps and d < best_dist:
                best, best_dist = s_rep, d
        return best

    # Representatives: D[rep_word] = (hidden_state, [equivalence_class_words])
    D: Dict[str, Tuple[torch.Tensor, List[str]]] = {}
    # Transitions over representative words: T[rep_word][symbol] = rep_word
    T: Dict[str, Dict[str, str]] = {}
    # Hidden cache
    hidden_cache: Dict[str, torch.Tensor] = {}

    # ----- initialization with ε (empty word)
    h_eps = hidden_of("")
    D[""] = (h_eps, [""])
    T[""] = {}

    # Queue holds only children of representatives (invariant)
    q: deque[str] = deque("".join([str(a)]) for a in alphabet)

    # ----- BFS until closure (fixpoint) or cap
    while q:
        if max_states is not None and len(D) >= max_states:
            break

        w = q.popleft()
        prefix = w[:-1] if len(w) > 0 else ""
        a = w[-1] if len(w) > 0 else ""
        assert prefix in D, "Invariant violation: expanding non-representative prefix."

        h = hidden_of(w)
        rep = find_rep(h)

        if rep is None:
            # ---- Case 3: New representative ----
            D[w] = (h, [w])
            # record transition from representative prefix
            T.setdefault(prefix, {})
            T[prefix][a] = w

            # ensure outgoing dict and enqueue its children
            T.setdefault(w, {})
            for b in alphabet:
                q.append(w + str(b))
        else:
            # ---- Merge (within ε) to existing representative ----
            T.setdefault(prefix, {})
            T[prefix][a] = rep
            D[rep][1].append(w)

    # Ensure every representative has all outgoing transitions recorded.
    # (Queue-empty condition should already guarantee this.)
    for s in D.keys():
        T.setdefault(s, {})

    # ----- Relabel representatives by integers and build DFA
    rep_words = list(D.keys())
    state_to_num = {s: i for i, s in enumerate(rep_words)}
    states = list(range(len(rep_words)))

    # Deterministic alphabet order for the FA
    alphabet_list = [str(a) for a in alphabet]

    # Transitions as str->str->str (rep labels)
    # Convert to ints using state_to_num and symbol order.
    transitions_int: Dict[str, Dict[str, str]] = {}
    for s, trans in T.items():
        transitions_int[s] = {}
        for sym, dst in trans.items():
            # sym must be string for the FA builder below
            transitions_int[s][str(sym)] = dst

    automaton = build_finite_automaton_from_states(
        states=rep_words,
        transitions=transitions_int,
        alphabet=alphabet_list,
        accepting_states=None,  # unknown from hidden-state view; leave None
    )
    return automaton


def build_finite_automaton_from_states(
    states: List[str],
    transitions: Dict[str, Dict[str, str]],
    alphabet: List[str],
    accepting_states: Optional[Set[str]] = None
) -> FiniteAutomaton:
    """
    Build a FiniteAutomaton from string-labeled states and transitions.

    Args
    ----
    states : list[str]
        Representative labels (ε, "0", "1", "01", ...). The first is the start.
    transitions : dict[state][symbol] = next_state
        Map over the given `states` and string symbols in `alphabet`.
    alphabet : list[str]
        Deterministic order of symbols (defines the integer encoding).
    accepting_states : set[str] | None
        Optional set of accepting reps; if None, leave FiniteAutomaton default.

    Returns
    -------
    FiniteAutomaton
    """
    state_to_int = {s: i for i, s in enumerate(states)}
    symbol_to_int = {sym: j for j, sym in enumerate(alphabet)}

    fa = FiniteAutomaton(num_states=len(states), alphabet_size=len(alphabet), seed=None)

    # Clear and fill transitions as integer pairs
    fa.transitions.clear()
    for s_name, trans_dict in transitions.items():
        s_int = state_to_int[s_name]
        for sym, next_s_name in trans_dict.items():
            if sym not in symbol_to_int:
                raise KeyError(f"Symbol {sym!r} not in provided alphabet.")
            a_int = symbol_to_int[sym]
            if next_s_name not in state_to_int:
                raise KeyError(f"Next state {next_s_name!r} not in provided states.")
            t_int = state_to_int[next_s_name]
            fa.transitions[(s_int, a_int)] = t_int

    if accepting_states is not None:
        fa.accepting_states = {state_to_int[s] for s in accepting_states}

    # Start at the first provided state (ε)
    fa.start_state = state_to_int[states[0]]

    return fa


# =====================
# Minimal demonstration
# =====================

def _demo():
    # Demo alphabet {'a','b'} and a random model shape (untrained).
    alphabet = ["a", "b"]
    symbol_to_idx = {a: i for i, a in enumerate(alphabet)}

    model = AutomatonRNN(input_size=1, hidden_size=32, output_size=7)

    dfa = build_dfa_until_fixpoint(
        model=model,
        symbol_to_idx=symbol_to_idx,
        alphabet_size=len(alphabet),
        eps=0.1,
        alphabet_symbols=alphabet,
        max_states=200,  # safety cap for demo
    )

    # If your FiniteAutomaton exposes a summary method:
    try:
        print(dfa.get_info())
    except Exception:
        print(f"DFA built with {dfa.num_states} states and alphabet_size={dfa.alphabet_size}.")

if __name__ == "__main__":
    _demo()