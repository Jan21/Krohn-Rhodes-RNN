import numpy as np
from typing import Dict, List, Tuple, Optional



class CascadeAutomaton:
    """A single automaton in a cascade that can depend on other automata states."""
    
    def __init__(self, num_states: int, alphabet_size: int = 2, 
                 dependency_states: Optional[int] = None, seed: int = None):
        self.num_states = num_states
        self.alphabet_size = alphabet_size
        self.dependency_states = dependency_states  # Number of states from dependent automaton
        self.alphabet = list(range(alphabet_size))
        
        if seed is not None:
            np.random.seed(seed)
        
        # Transition function depends on input symbol and possibly dependent automaton state
        self.transitions = {}
        self._generate_random_transitions()
        
        # Start state is always 0
        self.start_state = 0
    
    def _generate_random_transitions(self):
        """Generate random transition function."""
        for state in range(self.num_states):
            for symbol in self.alphabet:
                if self.dependency_states is None:
                    # Independent automaton: (state, symbol) -> next_state
                    key = (state, symbol)
                    next_state = np.random.randint(0, self.num_states)
                    self.transitions[key] = next_state
                else:
                    # Dependent automaton: (state, symbol, dep_state) -> next_state
                    for dep_state in range(self.dependency_states):
                        key = (state, symbol, dep_state)
                        next_state = np.random.randint(0, self.num_states)
                        self.transitions[key] = next_state
    
    def step(self, state: int, symbol: int, dep_state: Optional[int] = None) -> int:
        """Take one step in the automaton."""
        if self.dependency_states is None:
            key = (state, symbol)
        else:
            if dep_state is None:
                raise ValueError("Dependent automaton requires dependency state")
            key = (state, symbol, dep_state)
        
        return self.transitions.get(key, state)


class CascadeSystem:
    """A system of cascaded automata where each depends on the previous ones."""
    
    def __init__(self, num_automata: int = 5, states_per_automaton: int = 3, 
                 alphabet_size: int = 2, seed: int = None):
        self.num_automata = num_automata
        self.states_per_automaton = states_per_automaton
        self.alphabet_size = alphabet_size
        
        if seed is not None:
            np.random.seed(seed)
        
        self.automata = []
        self._create_cascade()
    
    def _create_cascade(self):
        """Create the cascade of automata."""
        for i in range(self.num_automata):
            if i == 0:
                # First automaton is independent
                automaton = CascadeAutomaton(
                    num_states=self.states_per_automaton,
                    alphabet_size=self.alphabet_size,
                    dependency_states=None,
                    seed=np.random.randint(0, 10000)
                )
            else:
                # Subsequent automata depend on the first automaton's state
                automaton = CascadeAutomaton(
                    num_states=self.states_per_automaton,
                    alphabet_size=self.alphabet_size,
                    dependency_states=self.states_per_automaton,  # All depend on first automaton
                    seed=np.random.randint(0, 10000)
                )
            self.automata.append(automaton)
    
    def run(self, sequence: List[int]) -> List[List[int]]:
        """
        Run all automata on a sequence.
        
        Returns:
            List of state sequences, one for each automaton
        """
        # Initialize current states
        current_states = [automaton.start_state for automaton in self.automata]
        all_states = [[] for _ in range(self.num_automata)]
        
        for symbol in sequence:
            new_states = []
            
            for i, automaton in enumerate(self.automata):
                if i == 0:
                    # First automaton is independent
                    next_state = automaton.step(current_states[i], symbol)
                else:
                    # Subsequent automata depend on first automaton's current state
                    next_state = automaton.step(current_states[i], symbol, current_states[i-1])
                
                new_states.append(next_state)
                all_states[i].append(next_state)
            
            current_states = new_states
        
        return all_states
    
    def get_info(self) -> Dict:
        """Get information about the cascade system."""
        info = {
            'num_automata': self.num_automata,
            'states_per_automaton': self.states_per_automaton,
            'alphabet_size': self.alphabet_size,
            'total_states': self.num_automata * self.states_per_automaton
        }
        
        # Add transition info for each automaton
        for i, automaton in enumerate(self.automata):
            info[f'automaton_{i}_transitions'] = dict(automaton.transitions)
        
        return info


def generate_cascade_system(num_automata: int = 5, states_per_automaton: int = 3, 
                          alphabet_size: int = 2, seed: int = None) -> CascadeSystem:
    """Generate a random cascade automaton system."""
    return CascadeSystem(
        num_automata=num_automata,
        states_per_automaton=states_per_automaton,
        alphabet_size=alphabet_size,
        seed=seed
    )