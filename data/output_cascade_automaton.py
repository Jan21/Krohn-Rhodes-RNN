import numpy as np
from typing import Dict, List, Tuple, Optional

class OutputCascadeAutomaton:
    """
    A Mealy-style automaton: given (state, input_symbol) it returns
    (next_state, output_symbol). The output_symbol will become the input
    to the next automaton in the cascade (if any).
    """
    def __init__(self, num_states: int, alphabet_size: int = 2, seed: Optional[int] = None):
        self.num_states = num_states
        self.alphabet_size = alphabet_size
        self.alphabet = list(range(alphabet_size))
        self.start_state = 0

        if seed is not None:
            np.random.seed(seed)

        # Transition: (state, input_symbol) -> next_state
        self.transitions: Dict[Tuple[int, int], int] = {}
        # Output map: (state, input_symbol) -> output_symbol (for the next layer)
        self.outputs: Dict[Tuple[int, int], int] = {}

        self._generate_random_mealy_rules()

    def _generate_random_mealy_rules(self):
        for s in range(self.num_states):
            for a in self.alphabet:
                self.transitions[(s, a)] = np.random.randint(0, self.num_states)
                self.outputs[(s, a)] = np.random.randint(0, self.alphabet_size)

    def step(self, state: int, in_symbol: int) -> Tuple[int, int]:
        """
        One Mealy step: returns (next_state, out_symbol).
        If a key is missing (shouldn't happen with our generation), stay in place and echo input.
        """
        key = (state, in_symbol)
        next_state = self.transitions.get(key, state)
        out_symbol = self.outputs.get(key, in_symbol % self.alphabet_size)
        return next_state, out_symbol


class OutputCascadeSystem:
    """
    A cascade of Mealy automata A0, A1, ..., A_{m-1} wired so that:
      - A0 receives the external input sequence x_t.
      - For i >= 1, Ai receives as input the output of A_{i-1} *at the same time step*.
    Updates are synchronous per time step: each layer uses its *current* state and input;
    outputs flow forward within the step; all next states are committed at the end of the step.
    """
    def __init__(
        self,
        num_automata: int = 5,
        states_per_automaton: int = 3,
        alphabet_size: int = 2,
        seed: Optional[int] = None,
    ):
        assert num_automata >= 1, "Need at least one automaton."

        self.num_automata = num_automata
        self.states_per_automaton = states_per_automaton
        self.alphabet_size = alphabet_size

        if seed is not None:
            np.random.seed(seed)

        self.automata: List[OutputCascadeAutomaton] = []
        for _ in range(num_automata):
            self.automata.append(
                OutputCascadeAutomaton(
                    num_states=states_per_automaton,
                    alphabet_size=alphabet_size,
                    seed=np.random.randint(0, 10_000) if seed is not None else None,
                )
            )

    def run(self, external_sequence: List[int]) -> Dict[str, List[List[int]]]:
        """
        Run the cascade on an external input sequence.

        Returns a dict with:
          - 'states': list per automaton of visited next states (length = len(external_sequence))
          - 'inputs': list per automaton of the inputs actually consumed at each step
                      (A0 gets the external sequence; Ai gets A_{i-1}'s outputs)
          - 'outputs': list per automaton of the outputs produced at each step
        """
        m = self.num_automata
        T = len(external_sequence)

        # Initialize states
        current_states = [a.start_state for a in self.automata]

        # Logs
        states_log: List[List[int]] = [[] for _ in range(m)]
        inputs_log: List[List[int]] = [[] for _ in range(m)]
        outputs_log: List[List[int]] = [[] for _ in range(m)]

        for t in range(T):
            # Prepare inputs for each layer at this time step
            layer_inputs = [0] * m
            layer_inputs[0] = external_sequence[t] % self.alphabet_size  # sanitize just in case

            # We compute next states and outputs sequentially, but
            # each layer uses the *current* state snapshot from the start of the step.
            next_states = [None] * m
            out_symbol_for_next_layer = None

            for i in range(m):
                if i > 0:
                    # Input to Ai is the output of A_{i-1} from this same step
                    layer_inputs[i] = out_symbol_for_next_layer

                s = current_states[i]
                a = layer_inputs[i]
                ns, out_sym = self.automata[i].step(s, a)

                next_states[i] = ns
                out_symbol_for_next_layer = out_sym  # becomes input for the next layer

                # Logs
                states_log[i].append(ns)
                inputs_log[i].append(a)
                outputs_log[i].append(out_sym)

            # Commit state update synchronously
            current_states = next_states  # type: ignore

        return {"states": states_log, "inputs": inputs_log, "outputs": outputs_log}

    def get_info(self) -> Dict:
        info = {
            "num_automata": self.num_automata,
            "states_per_automaton": self.states_per_automaton,
            "alphabet_size": self.alphabet_size,
            "total_states": self.num_automata * self.states_per_automaton,
        }
        for i, a in enumerate(self.automata):
            info[f"automaton_{i}_transitions"] = dict(a.transitions)
            info[f"automaton_{i}_outputs"] = dict(a.outputs)
        return info


def generate_output_cascade_system(
    num_automata: int = 5,
    states_per_automaton: int = 3,
    alphabet_size: int = 2,
    seed: Optional[int] = None,
) -> OutputCascadeSystem:
    return OutputCascadeSystem(
        num_automata=num_automata,
        states_per_automaton=states_per_automaton,
        alphabet_size=alphabet_size,
        seed=seed,
    )