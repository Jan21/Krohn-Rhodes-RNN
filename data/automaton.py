import numpy as np
import torch
from typing import Dict, List, Tuple, Set


class FiniteAutomaton:
    """A finite automaton with alphabet {0, 1, 2, 3, 4}."""
    
    def __init__(self, num_states: int, alphabet_size: int = 5, seed: int = None):
        self.num_states = num_states
        self.alphabet_size = alphabet_size
        self.alphabet = list(range(alphabet_size))
        
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


def generate_random_automaton(num_states: int = 10, seed: int = None) -> FiniteAutomaton:
    """Generate a random finite automaton with the specified number of states."""
    return FiniteAutomaton(num_states=num_states, seed=seed)