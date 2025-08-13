import numpy as np
import torch
import graphviz
from typing import Dict, List, Tuple, Set


class FiniteAutomaton:
    """A finite automaton with alphabet {0, 1, 2, 3, 4}."""
    
    def __init__(self, num_states: int, alphabet_size: int = 5, seed: int = None, name : str = "Automaton"):
        self.num_states = num_states
        self.alphabet_size = alphabet_size
        self.alphabet = list(range(alphabet_size))
        self.name = name
        
        if seed is not None:
            np.random.seed(seed)
        
        # Transition function: (state, symbol) -> state
        self.transitions = {}
        self._generate_random_transitions()
        
        # Start state is always 0
        self.start_state = 0
        
        # Random set of accepting states (at least one, at most half)
        num_accepting = np.random.randint(1, max(2, num_states // 2 + 1))
        self.accepting_states = set(np.random.choice(
            num_states, size=num_accepting, replace=False
        ))
    
    def _generate_random_transitions(self):
        """Generate random transition function."""
        for state in range(self.num_states):
            for symbol in self.alphabet:
                # Random next state
                next_state = np.random.randint(0, self.num_states)
                self.transitions[(state, symbol)] = next_state
    
    def step(self, state: int, symbol: int) -> int:
        """Take one step in the automaton."""
        return self.transitions.get((state, symbol), state)
    
    def run(self, sequence: List[int]) -> Tuple[List[int], List[bool]]:
        """
        Run the automaton on a sequence.
        
        Returns:
            states: List of states after each symbol
            accepts: List of whether each state is accepting
        """
        current_state = self.start_state
        states = []
        accepts = []
        
        for symbol in sequence:
            current_state = self.step(current_state, symbol)
            states.append(current_state)
            accepts.append(current_state in self.accepting_states)
        
        return states, accepts
    
    def accepts_sequence(self, sequence: List[int]) -> bool:
        """Check if the automaton accepts the entire sequence."""
        states, accepts = self.run(sequence)
        return accepts[-1] if accepts else False
    
    def get_info(self) -> Dict:
        """Get information about the automaton."""
        return {
            'num_states': self.num_states,
            'alphabet_size': self.alphabet_size,
            'start_state': self.start_state,
            'accepting_states': list(self.accepting_states),
            'transitions': dict(self.transitions)
        }
    
    def to_dot(self, name: str = "Automaton") -> str:
        """
        Export the automaton to a Graphviz DOT string.

        - States are named s0, s1, ...
        - Start arrow drawn from an invisible node
        - Accepting states in double circles
        - For deterministic transitions, multiple symbols that go from u to v
        are merged into a single edge label "a,b,c".
        """
        # Collect labels per (u, v)
        edge_labels = {}
        for (u, a), v in self.transitions.items():
            edge_labels.setdefault((u, v), []).append(a)

        lines = []
        lines.append(f'digraph {name} {{')
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=circle];')

        # Accepting states as doublecircle
        if self.accepting_states:
            acc = " ".join(f"s{q}" for q in sorted(self.accepting_states))
            lines.append(f'  node [shape=doublecircle]; {acc};')
            lines.append('  node [shape=circle];')  # reset for the rest

        # Invisible start arrow
        lines.append('  "" [shape=none, width=0, height=0, label=""];')
        lines.append(f'  "" -> s{self.start_state};')

        # Ensure all states appear even if isolated
        for q in range(self.num_states):
            lines.append(f'  s{q} [label="{q}"];')

        # Edges with merged labels
        for (u, v), syms in edge_labels.items():
            # Sort labels numerically then join
            label = ",".join(str(x) for x in sorted(syms))
            lines.append(f'  s{u} -> s{v} [label="{label}"];')

        lines.append('}')
        return "\n".join(lines)


def generate_random_automaton(num_states: int = 10, seed: int = None) -> FiniteAutomaton:
    """Generate a random finite automaton with the specified number of states."""
    return FiniteAutomaton(num_states=num_states, seed=seed)

def render(automaton:FiniteAutomaton):
    g = graphviz.Source(fa.to_dot("DFA"))
    g.render("graphs/"+automaton.name, format="png", cleanup=True)

from typing import Dict, List, Set, Tuple, Optional
from collections import deque, defaultdict

def minimize_dfa_hopcroft(
    fa: "FiniteAutomaton",
    *,
    add_sink_if_needed: bool = True,
    trim_unreachable: bool = True,
) -> Tuple["FiniteAutomaton", Dict[int, Optional[int]], List[Set[int]]]:
    """
    Minimize a deterministic finite automaton using Hopcroft's algorithm,
    adapted to the given `FiniteAutomaton` API.

    Parameters
    ----------
    fa : FiniteAutomaton
        Deterministic FA with attributes:
        - num_states : int
        - alphabet_size : int   (alphabet is assumed {0,...,k-1})
        - transitions : Dict[(state:int, symbol:int), state:int]
        - start_state : int
        - accepting_states : Set[int]
    add_sink_if_needed : bool
        If True, make δ total by adding an explicit sink state that self-loops
        and receives all missing transitions. If False, missing transitions are
        treated as self-loops on the source state (matches `step()` fallback).
    trim_unreachable : bool
        If True, restrict to states reachable from the start (yields the unique
        minimal DFA for the language recognized by `fa`).

    Returns
    -------
    min_fa : FiniteAutomaton
        The minimized DFA (same alphabet size).
    old_to_new : Dict[int, Optional[int]]
        Maps original state -> new minimized state id; unreachable states map to None.
    blocks : List[Set[int]]
        Partition blocks (in the order used to build states of `min_fa`);
        each block is a set of original state ids merged into that minimized state.

    Correctness sketch
    ------------------
    1) We first totalize δ (Hopcroft assumes a total transition function).
    2) Optionally trim to the reachable subgraph from the start (standard to get
       the unique minimal DFA up to isomorphism).
    3) Run partition refinement: initialize P = {F, Q\\F} (drop empties), then
       repeatedly split blocks using preimages X_a = δ^{-1}(A) for A in the worklist
       until no block is splittable. The resulting blocks are Myhill–Nerode classes.
    4) Build the quotient automaton by choosing any representative in each block.

    Complexity: O(|Σ| |Q| log |Q|) with the “add the smaller piece to the worklist” heuristic.
    """
    k = fa.alphabet_size
    Sigma = list(range(k))

    # ---------- 0) Build total δ (add sink if needed or treat missing as self-loop) ----------
    n0 = fa.num_states
    delta: List[List[int]] = [[-1] * k for _ in range(n0)]
    for (q, a), t in fa.transitions.items():
        delta[q][a] = t

    missing = any(delta[q][a] < 0 for q in range(n0) for a in Sigma)
    sink = None
    if add_sink_if_needed and missing:
        sink = n0
        delta.append([sink] * k)              # sink loops to itself
        for q in range(n0):
            for a in Sigma:
                if delta[q][a] < 0:
                    delta[q][a] = sink
        nT = n0 + 1
    else:
        # Interpret missing as self-loop (consistent with FiniteAutomaton.step())
        for q in range(n0):
            for a in Sigma:
                if delta[q][a] < 0:
                    delta[q][a] = q
        nT = n0

    accepting0 = set(fa.accepting_states)
    if sink is not None:
        accepting0.discard(sink)  # sink is non-accepting

    # ---------- 0.5) (Optional) remove unreachable states ----------
    if trim_unreachable:
        start = fa.start_state
        visited: Set[int] = set()
        dq = deque([start])
        visited.add(start)
        while dq:
            q = dq.popleft()
            for a in Sigma:
                t = delta[q][a]
                if t not in visited:
                    visited.add(t)
                    dq.append(t)

        # Map reachable original ids -> compact indices 0..m-1
        reach = sorted(visited)
        idmap: Dict[int, int] = {q: i for i, q in enumerate(reach)}
        m = len(reach)

        # Restrict δ and F to reachable
        delta_R: List[List[int]] = [[0] * k for _ in range(m)]
        for q in reach:
            iq = idmap[q]
            for a in Sigma:
                t = delta[q][a]
                delta_R[iq][a] = idmap[t]  # t must be reachable by construction

        F_R: Set[int] = {idmap[q] for q in accepting0 if q in idmap}
        start_R = idmap[start]

        # Keep a back-pointer to original ids for reporting blocks later
        back_to_old: List[int] = reach
    else:
        # No trimming: work with 0..nT-1
        m = nT
        delta_R = [row[:] for row in delta]
        F_R = {q for q in range(nT) if q in accepting0}
        start_R = fa.start_state
        back_to_old = list(range(nT))

    # ---------- 1) Precompute predecessors: pre[a][v] = { q | δ(q,a)=v } ----------
    pre: List[List[Set[int]]] = [ [set() for _ in range(m)] for _ in Sigma ]
    for q in range(m):
        for a in Sigma:
            v = delta_R[q][a]
            pre[a][v].add(q)

    # ---------- 2) Initialize partition P and worklist W ----------
    Qm = set(range(m))
    Fblock = set(F_R)
    NFblock = Qm - Fblock

    P: List[Set[int]] = []
    if Fblock:
        P.append(Fblock)
    if NFblock:
        P.append(NFblock)

    W: List[Set[int]] = []
    if Fblock and NFblock:
        W.append(Fblock if len(Fblock) <= len(NFblock) else NFblock)
    elif Fblock:
        W.append(Fblock)
    elif NFblock:
        W.append(NFblock)

    # ---------- 3) Hopcroft refinement loop ----------
    while W:
        A = W.pop()
        for a in Sigma:
            # X = δ^{-1}(A) on symbol a
            X: Set[int] = set()
            for v in A:
                X |= pre[a][v]
            if not X:
                continue

            newP: List[Set[int]] = []
            for Y in P:
                iY = Y & X
                dY = Y - X
                if iY and dY:
                    newP.extend([iY, dY])
                    if Y in W:
                        W.remove(Y)
                        W.append(iY)
                        W.append(dY)
                    else:
                        W.append(iY if len(iY) <= len(dY) else dY)
                else:
                    newP.append(Y)
            P = newP

    # Blocks in a fixed order (as created)
    blocks_R: List[Set[int]] = [B for B in P if B]

    # ---------- 4) Build minimized DFA (quotient by blocks) ----------
    # Map reduced-state -> new-state
    red_to_new: Dict[int, int] = {}
    for i, B in enumerate(blocks_R):
        for q in B:
            red_to_new[q] = i

    new_n = len(blocks_R)
    new_k = k

    # Transitions: pick any representative per block
    new_transitions: Dict[Tuple[int, int], int] = {}
    for i, B in enumerate(blocks_R):
        qrep = next(iter(B))
        for a in Sigma:
            t = delta_R[qrep][a]
            j = red_to_new[t]
            new_transitions[(i, a)] = j

    # New start and accepting blocks
    new_start = red_to_new[start_R]
    new_accepting: Set[int] = set()
    for i, B in enumerate(blocks_R):
        if any(q in F_R for q in B):
            new_accepting.add(i)

    # ---------- 5) Construct a FiniteAutomaton instance with these parameters ----------
    min_fa = FiniteAutomaton(num_states=new_n, alphabet_size=new_k, seed=None, name=f"{fa.name}_min")
    min_fa.transitions.clear()
    min_fa.transitions.update(new_transitions)
    min_fa.start_state = new_start
    min_fa.accepting_states = set(new_accepting)

    # Old(original) -> New mapping (unreachable map to None)
    old_to_new: Dict[int, Optional[int]] = {old: None for old in range(fa.num_states)}
    # red-state q corresponds to original old = back_to_old[q]
    for q_red in range(m):
        old = back_to_old[q_red]
        old_to_new[old] = red_to_new[q_red]

    # Report blocks in terms of original state ids
    blocks_old: List[Set[int]] = [{back_to_old[q] for q in B} for B in blocks_R]

    return min_fa, old_to_new, blocks_old


if __name__ == "__main__":
    fa = FiniteAutomaton(2, alphabet_size=2,seed=2)
    print(fa.get_info())
    render(fa)
    