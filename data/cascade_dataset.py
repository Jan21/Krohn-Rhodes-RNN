import torch
from torch.utils.data import Dataset
import numpy as np
from .cascade_automaton import CascadeSystem


class CascadeDataset(Dataset):
    """Dataset for training RNNs on cascade automaton systems."""
    
    def __init__(self, cascade_system: CascadeSystem, num_samples=1000, seq_length=10, seed=None):
        self.cascade_system = cascade_system
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_automata = cascade_system.num_automata
        self.states_per_automaton = cascade_system.states_per_automaton
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.sequences, self.labels = self._generate_data()
    
    def _generate_data(self):
        sequences = []
        labels = []
        
        for _ in range(self.num_samples):
            # Generate random binary sequence
            sequence = torch.randint(0, 2, (self.seq_length,), dtype=torch.float32)
            
            # Run cascade system on sequence to get states for all automata
            all_states = self.cascade_system.run(sequence.int().tolist())
            
            # Stack states from all automata: shape (seq_length, num_automata)
            state_tensor = torch.tensor(all_states, dtype=torch.long).T  # Transpose to get correct shape
            
            sequences.append(sequence.unsqueeze(-1))  # Shape: (seq_length, 1)
            labels.append(state_tensor)  # Shape: (seq_length, num_automata)
        
        return torch.stack(sequences), torch.stack(labels)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]