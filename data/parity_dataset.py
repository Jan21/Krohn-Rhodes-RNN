import torch
from torch.utils.data import Dataset
import numpy as np


class ParityDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=10, seed=None):
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
            sequence = torch.randint(0, 2, (self.seq_length,), dtype=torch.float32)
            
            running_parity = torch.zeros(self.seq_length, dtype=torch.long)
            ones_count = 0
            
            for i in range(self.seq_length):
                if sequence[i] == 1:
                    ones_count += 1
                running_parity[i] = ones_count % 2
            
            sequences.append(sequence.unsqueeze(-1))
            labels.append(running_parity.unsqueeze(-1))
        
        return torch.stack(sequences), torch.stack(labels)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]