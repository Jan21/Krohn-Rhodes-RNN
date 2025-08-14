import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from .automaton import FiniteAutomaton
import graphviz



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
                 alphabet_size: int = 2, seed: int = None,name:str="CascadeSystem"):
        self.num_automata = num_automata
        self.states_per_automaton = states_per_automaton
        self.alphabet_size = alphabet_size
        self.name = name
        
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


def render_fa(automaton: FiniteAutomaton, outdir: str = "graphs"):
    g = graphviz.Source(automaton.to_dot(automaton.name))
    g.render(f"{outdir}/{automaton.name}", format="png", cleanup=True)


# ---------- Cascade rendering ----------
def cascade_to_dot(
    system: CascadeSystem,
    name: str = "Cascade",
    sequence: Optional[List[int]] = None,
) -> str:
    """
    Export the cascade system to a Graphviz DOT string.

    Layout:
      - one cluster per automaton (cluster_0, cluster_1, ...)
      - node names: A{i}_s{q}, labels "i:q"
      - edges:
          * i = 0:          (state, symbol) -> next_state    labeled "a"
          * i >= 1 (dep):   (state, symbol, dep_state) ->    labeled "a|d=dep"
        Multiple labels from u->v are merged: "0|d=1, 1|d=2, ..."

    If `sequence` is provided, the unique path induced by `sequence` is highlighted:
      - visited nodes: filled with lightgray
      - used edges: penwidth=2
    """
    lines: List[str] = []
    lines.append(f'digraph {name} {{')
    lines.append('  rankdir=LR;')
    lines.append('  fontsize=12;')
    lines.append('  node [shape=circle, fontsize=12];')

    # --- If highlighting, simulate the run and collect edges/nodes used ---
    highlight_nodes: List[Set[Tuple[int, int]]] = [set() for _ in range(system.num_automata)]
    highlight_edges: List[Set[Tuple[int, int]]] = [set() for _ in range(system.num_automata)]
    if sequence is not None:
        current = [automaton.start_state for automaton in system.automata]
        for a in sequence:
            new_states = []
            for i, automaton in enumerate(system.automata):
                if i == 0:
                    key = (current[i], a)
                    nxt = automaton.transitions[key]
                else:
                    key = (current[i], a, current[i-1])
                    nxt = automaton.transitions[key]
                # record edge and nodes
                highlight_edges[i].add((current[i], nxt))
                highlight_nodes[i].add((i, current[i]))
                highlight_nodes[i].add((i, nxt))
                new_states.append(nxt)
            current = new_states
        # also include all start nodes
        for i, automaton in enumerate(system.automata):
            highlight_nodes[i].add((i, automaton.start_state))

    # --- Build clusters, nodes, and edges for each automaton ---
    for i, automaton in enumerate(system.automata):
        lines.append(f'  subgraph cluster_{i} {{')
        lines.append(f'    label="Automaton {i}{" (dep on "+str(i-1)+")" if i>0 else ""}";')
        lines.append('    style=rounded;')

        # Ensure all nodes appear
        for q in range(automaton.num_states):
            node_name = f'A{i}_s{q}'
            label = f'{i}:{q}'
            if (i, q) in highlight_nodes[i]:
                lines.append(f'    {node_name} [label="{label}", style=filled, fillcolor="lightgray"];')
            else:
                lines.append(f'    {node_name} [label="{label}"];')

        # Invisible start arrow for each automaton
        lines.append(f'    "start_{i}" [shape=none, width=0, height=0, label=""];')
        lines.append(f'    "start_{i}" -> A{i}_s{automaton.start_state};')

        # Collect labels per (u, v)
        edge_labels: Dict[Tuple[int, int], List[str]] = {}
        if automaton.dependency_states is None:
            # independent: transitions keys are (state, symbol)
            for (u, a), v in automaton.transitions.items():
                edge_labels.setdefault((u, v), []).append(str(a))
        else:
            # dependent: keys are (state, symbol, dep)
            for (u, a, dep), v in automaton.transitions.items():
                edge_labels.setdefault((u, v), []).append(f"{a}|d={dep}")

        # Emit merged edges
        for (u, v), labels in edge_labels.items():
            node_u = f'A{i}_s{u}'
            node_v = f'A{i}_s{v}'
            lbl = ",".join(sorted(labels, key=lambda s: [int(x) if x.isdigit() else x for x in s.replace("|d="," ").replace("|"," ").replace(":"," ").split()]))
            if (u, v) in highlight_edges[i]:
                lines.append(f'    {node_u} -> {node_v} [label="{lbl}", penwidth=2];')
            else:
                lines.append(f'    {node_u} -> {node_v} [label="{lbl}"];')

        lines.append('  }')  # end cluster

    lines.append('}')
    return "\n".join(lines)


def render_cascade(system: CascadeSystem, outpath: str = "prints/graphs", sequence: Optional[List[int]] = None):
    """
    Render the cascade system to PNG. If `sequence` is provided, highlight the run.
    """
    dot = cascade_to_dot(system, name=system.__class__.__name__, sequence=sequence)
    g = graphviz.Source(dot)
    g.render(outpath+"/"+system.name, format="png", cleanup=True)

# adapters.py
class OutputCascadeAdapter:
    """
    Wraps OutputCascadeSystem so it matches the old interface:
      - run(sequence) -> List[List[int]]  (states only, per automaton)
      - get_info() passed through
    """
    def __init__(self, sys):
        self.sys = sys
        # Mirror the attributes used elsewhere
        self.num_automata = sys.num_automata
        self.states_per_automaton = sys.states_per_automaton
        self.alphabet_size = sys.alphabet_size

    def run(self, sequence):
        result = self.sys.run(sequence)
        # Old code expects only the states
        return result["states"]

    def get_info(self):
        return self.sys.get_info()