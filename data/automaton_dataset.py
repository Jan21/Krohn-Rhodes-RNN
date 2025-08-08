import torch
from torch.utils.data import Dataset
import numpy as np
from .automaton import FiniteAutomaton


class AutomatonDataset(Dataset):
    """Dataset for training RNNs on arbitrary finite automata."""
    
    def __init__(self, automaton: FiniteAutomaton, num_samples=1000, seq_length=10, seed=None):
        self.automaton = automaton
        self.num_samples = num_samples
        self.seq_length = seq_length
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.sequences, self.labels = self._generate_data()
    
    def _generate_data(self):
        sequences = []
        labels = []
        
        for _ in range(self.num_samples):
            # Generate random sequence from alphabet [0,1,2,3,4]
            sequence = torch.randint(0, 5, (self.seq_length,), dtype=torch.float32)
            
            # Run automaton on sequence to get step-by-step state labels
            states, _ = self.automaton.run(sequence.int().tolist())
            
            # Convert to tensor (predict the actual state ID)
            running_labels = torch.tensor(states, dtype=torch.long)
            
            sequences.append(sequence.unsqueeze(-1))
            labels.append(running_labels)
        
        return torch.stack(sequences), torch.stack(labels)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]