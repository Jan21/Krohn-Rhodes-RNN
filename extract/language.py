"""
RNN-to-DFA extraction (adapted for AutomatonRNN with input_size=1
and the provided FiniteAutomaton class).

This file is self-contained:
- includes the AutomatonRNN and FiniteAutomaton classes (from your friend),
- utilities to encode inputs, get hidden states,
- FiniteLanguage structure + builders,
- and a build_dfa_from_language() that clusters hidden states with an ε-threshold.

Author: ChatGPT (adapted per user's request)
"""

from __future__ import annotations
from typing import Any, Dict, Hashable, List, Optional, Sequence, Set, Tuple

import sys
import os

import random
import numpy as np
import torch
import torch.nn as nn

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from models.automaton_rnn import AutomatonRNN
from data.automaton import FiniteAutomaton


# =====================
# Utility / Converters
# =====================

def random_words(A: Sequence[Any], n: int, k: int) -> List[List[Any]]:
    """
    Generate k random words over alphabet A, each of length n.
    """
    if not A:
        raise ValueError("Alphabet A must be non-empty.")
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative integers.")
    words = [random.choices(list(A), k=n) for _ in range(k)]
    return [''.join(map(str, w)) for w in words]


# def _normalize_transitions_dict(transitions_in: Dict[Any, Any]) -> Dict[Tuple[Hashable, Hashable], Any]:
#     """
#     Accepts transition keys either as tuples (state, symbol) or strings "state,symbol".
#     Returns a dict with tuple keys (state_label, symbol_label).
#     """
#     out: Dict[Tuple[Hashable, Hashable], Any] = {}
#     for k, v in transitions_in.items():
#         if isinstance(k, tuple) and len(k) == 2:
#             s, a = k
#         elif isinstance(k, str):
#             parts = k.split(",")
#             if len(parts) != 2:
#                 raise ValueError(f"Transition key '{k}' is not of the form 'state,symbol'.")
#             s, a = parts[0].strip(), parts[1].strip()
#         else:
#             raise TypeError(f"Unsupported transition key type: {type(k)} (key={k!r}).")
#         out[(s, a)] = v
#     return out


# def finite_automaton_from_dict(data: Dict[str, Any]):
#     """
#     Build a FiniteAutomaton from a dictionary that may use arbitrary labels.

#     Returns:
#       (fa, state_to_int, symbol_to_int, int_to_state, int_to_symbol)
#     """
#     if "states" not in data or "alphabet" not in data or "transitions" not in data:
#         raise KeyError("data must contain 'states', 'alphabet', and 'transitions'.")

#     states_labels = list(data["states"])
#     alphabet_labels = list(data["alphabet"])
#     transitions_in = _normalize_transitions_dict(data["transitions"])

#     init_label = data.get("initial_state", data.get("start_state", None))
#     if init_label is None:
#         raise KeyError("data must contain 'initial_state' or 'start_state'.")

#     accepting_labels = set(data.get("accepting_states", []))

#     state_to_int = {s: i for i, s in enumerate(states_labels)}
#     int_to_state = {i: s for s, i in state_to_int.items()}

#     symbol_to_int = {a: j for j, a in enumerate(alphabet_labels)}
#     int_to_symbol = {j: a for a, j in symbol_to_int.items()}

#     for (s, a), t in transitions_in.items():
#         if s not in state_to_int:
#             raise ValueError(f"Transition references undeclared state label: {s!r}")
#         if a not in symbol_to_int:
#             raise ValueError(f"Transition references undeclared symbol label: {a!r}")
#         if t not in state_to_int:
#             raise ValueError(f"Transition targets undeclared state label: {t!r}")
#     if init_label not in state_to_int:
#         raise ValueError(f"Initial state label {init_label!r} not in 'states'.")
#     for acc in accepting_labels:
#         if acc not in state_to_int:
#             raise ValueError(f"Accepting state label {acc!r} not in 'states'.")

#     transitions_int: Dict[Tuple[int, int], int] = {}
#     for (s, a), t in transitions_in.items():
#         si = state_to_int[s]
#         ai = symbol_to_int[a]
#         ti = state_to_int[t]
#         transitions_int[(si, ai)] = ti

#     fa = FiniteAutomaton(num_states=len(states_labels), alphabet_size=len(alphabet_labels), seed=0)
#     fa.transitions = transitions_int
#     fa.start_state = state_to_int[init_label]
#     fa.accepting_states = {state_to_int[s] for s in accepting_labels}
#     return fa, state_to_int, symbol_to_int, int_to_state, int_to_symbol


# =====================
# Finite language type
# =====================

class FiniteLanguage:
    """
    A structured finite language represented as levels, where level n contains
    all words of length n and each word in level n+1 extends a word from level n.
    """
    def __init__(self, levels: List[Set[str]], alphabet: Set[str]):
        self.levels = levels
        self.alphabet = alphabet
        self._check_well_formedness()

    def _check_well_formedness(self):
        """
        Verifies consistency:
        1. Level 0 contains only the empty string.
        2. Every word in level n extends to some word in level n+1.
        3. Every word in level n+1 has its length-(n) prefix in level n.
        """
        L0 = self.levels[0]
        if len(L0) > 1 or "" not in L0:
            raise ValueError("Level 0 must contain only the empty string.")

        for n in range(len(self.levels) - 1):
            Ln = self.levels[n]
            Ln1 = self.levels[n + 1]

            # Forward extension
            for w in Ln:
                if not any((w + a) in Ln1 for a in self.alphabet):
                    raise ValueError(
                        f"Word '{w}' in L_{n} does not extend to any word in L_{n+1}."
                    )
            # Backward compatibility
            for w in Ln1:
                prefix = w[:-1]
                if prefix not in Ln:
                    raise ValueError(
                        f"Word '{w}' in L_{n+1} has prefix '{prefix}' not in L_{n}."
                    )

    def __repr__(self) -> str:
        return f"FiniteLanguage(levels={self.levels})"


def finite_language_from_sequences(strings: List[str]) -> FiniteLanguage:
    """
    Construct a FiniteLanguage from equal-length strings, collecting all prefixes.
    """
    assert strings, "Input list is empty."
    length = len(strings[0])
    assert all(len(s) == length for s in strings), "All strings must have the same length."

    levels: List[Set[str]] = [set() for _ in range(length + 1)]
    alphabet: Set[str] = set()

    for s in strings:
        for i in range(len(s) + 1):
            prefix = s[:i]
            levels[i].add(prefix)
        alphabet.update(s)

    # Ensure the empty word is present
    levels[0].add("")
    return FiniteLanguage(levels, alphabet)


# =======================================
# AutomatonRNN input encoding + extraction
# =======================================

@torch.no_grad()
def _symbols_to_scalar_sequence(
    symbols: Sequence[int],
    alphabet_size: int
) -> torch.Tensor:
    """
    Returns a float tensor of shape (1, T, 1) for any T >= 0.
    """
    T = len(symbols)
    if T == 0:
        # Explicit empty sequence with correct rank
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
    Return the hidden state h_T after reading 'word'. Handles empty word safely.
    """
    if device is None:
        device = next(model.parameters()).device

    ints = [symbol_to_idx[sym] for sym in word]

    # Handle empty word BEFORE touching the RNN
    if len(ints) == 0:
        h0 = torch.zeros(model.hidden_size, device=device)  # (H,)
        return h0.clone()

    x = _symbols_to_scalar_sequence(ints, alphabet_size).to(device)  # (1, T, 1)
    model.eval()
    h0 = torch.zeros(1, 1, model.hidden_size, device=device)  # (num_layers=1, B=1, H)
    rnn_out, _ = model.rnn(x, h0)  # (1, T, H)
    h_T = rnn_out[0, -1, :].clone()
    return h_T


# ==================================
# Hidden-state DFA "extraction" core
# ==================================

def build_dfa_from_language(
    language: FiniteLanguage,
    model: AutomatonRNN,
    symbol_to_idx: Dict[str, int],
    alphabet_size: int,
    eps: float
) -> Dict[str, Tuple[torch.Tensor, List[str], Dict[str, Dict[str, str]]]]:
    """
    Build a hidden-state-based DFA structure from a finite language and AutomatonRNN.

    For each word w in the language:
    - Compute hidden state h_T via AutomatonRNN.
    - If the prefix of w is already recorded in L, add w to L.
    - Else, if h_T is within eps of some existing representative state in D, link to it.
    - Otherwise, create a new representative state keyed by w.

    Returns:
        D: dict mapping representative word s to a tuple
           (hidden_state_of_s, associated_words_list, transitions_snapshot_dict)
    """
    D: Dict[str, Tuple[torch.Tensor, List[str]]] = {}
    T: Dict[str, Dict[str, str]] = {}
    L: List[str] = []
    hidden_cache: Dict[str, torch.Tensor] = {}

    for level in language.levels:
        for w in sorted(level):
            prefix = w[:-1] if w else ''
            if w not in hidden_cache:
                h = get_hidden_state_automaton_rnn(
                    model=model,
                    word=list(w),
                    symbol_to_idx=symbol_to_idx,
                    alphabet_size=alphabet_size
                )
                hidden_cache[w] = h
            else:
                h = hidden_cache[w]

            # Case 1: prefix in L: every extension 
            if prefix in L:
                L.append(w)
                continue

            # Case 2: merge by ε-distance
            found = False
            for s_rep, (h_rep, _) in D.items():
                if torch.norm(h - h_rep).item() < eps:
                    L.append(w)
                    D[s_rep][1].append(w)
                    if prefix not in T:
                        T[prefix] = {}
                    if w:
                        T[prefix][w[-1]] = s_rep
                    found = True
                    break

            if found:
                continue

            # Case 3: new representative
            D[w] = (h, [])
            if prefix not in T:
                T[prefix] = {}
            if w:
                T[prefix][w[-1]] = w

    states = list(D.keys())
    state_to_num = {state: i for i, state in enumerate(states)}
    states = [i for i,_ in enumerate(states)]

    # Step 2: Build the relabeled transitions
    new_transitions = {}
    for state, trans in T.items():
        new_state = state_to_num[state]  # number for this state
        new_transitions[new_state] = {sym: state_to_num[next_state]
                                    for sym, next_state in trans.items()}
    alphabet = {key for inner_dict in new_transitions.values() for key in inner_dict.keys()}
    automaton = build_finite_automaton_from_states(states=states,
                                                   transitions=new_transitions,
                                                   alphabet=alphabet)
    return automaton


def build_finite_automaton_from_states(
    states: List[str],
    transitions: Dict[str, Dict[str, str]],
    alphabet: List[str],
    accepting_states: Set[str] = None
) -> FiniteAutomaton:
    """
    Build a FiniteAutomaton from a list of states and a string-keyed transition map.

    Args:
        states: list of state names (in desired numbering order).
        transitions: dict mapping state_name -> {symbol: next_state_name}.
        alphabet: list of symbols (same order as symbol_to_int mapping).
        accepting_states: optional set of state names considered accepting.

    Returns:
        A FiniteAutomaton with transitions set exactly as provided.
    """
    # Map states and symbols to ints
    state_to_int = {s: i for i, s in enumerate(states)}
    symbol_to_int = {sym: j for j, sym in enumerate(alphabet)}

    fa = FiniteAutomaton(num_states=len(states), alphabet_size=len(alphabet), seed=None)
    # Overwrite transitions to match the provided mapping
    fa.transitions.clear()
    for s_name, trans_dict in transitions.items():
        s_int = state_to_int[s_name]
        for sym, next_s_name in trans_dict.items():
            a_int = symbol_to_int[sym]
            t_int = state_to_int[next_s_name]
            fa.transitions[(s_int, a_int)] = t_int

    # Overwrite accepting states if provided
    if accepting_states is not None:
        fa.accepting_states = {state_to_int[s] for s in accepting_states}

    # Start state is assumed to be states[0]
    fa.start_state = state_to_int[states[0]]

    return fa


# =====================
# Minimal demonstration
# =====================

def _demo():
    # Build a tiny language from two equal-length strings
    strings = random_words(A=["a","b"],n=12,k=36)
    lang = finite_language_from_sequences(strings)

    # Build symbol_to_idx from the discovered alphabet
    # (deterministic order for reproducibility)
    alphabet_sorted = sorted(list(lang.alphabet))
    symbol_to_idx = {a: i for i, a in enumerate(alphabet_sorted)}
    alphabet_size = len(symbol_to_idx)

    # Instantiate the RNN
    model = AutomatonRNN(input_size=1, hidden_size=32, output_size=7)

    # Run the extraction
    hidden_automaton = build_dfa_from_language(
        language=lang,
        model=model,
        symbol_to_idx=symbol_to_idx,
        alphabet_size=alphabet_size,
        eps=0.1
    )

    print(hidden_automaton.get_info())

if __name__ == "__main__":
    _demo()