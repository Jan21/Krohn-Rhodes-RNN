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
    g.render("prints/graphs/"+automaton.name, format="png", cleanup=True)




if __name__ == "__main__":
    fa = FiniteAutomaton(2, alphabet_size=2,seed=2)
    print(fa.get_info())
    render(fa)
    