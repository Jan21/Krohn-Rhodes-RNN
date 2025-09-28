import torch
import torch.nn as nn
import torch.nn.functional as F


class CascadeRNNBaseline(nn.Module):
    """Baseline LSTM model for predicting cascade automaton states."""
    
    def __init__(self, input_size=1, hidden_size=100, num_automata=5, states_per_automaton=3, attention_dim=64):
        super(CascadeRNNBaseline, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_automata = num_automata
        self.states_per_automaton = states_per_automaton
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_size, 10)
        
        # Single LSTM for processing the sequence
        self.lstm = nn.LSTM(
            input_size=10,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Separate readout head for each automaton
        self.readout_heads = nn.ModuleList([
            nn.Linear(hidden_size, states_per_automaton) 
            for _ in range(num_automata)
        ])
        
    def forward(self, x):
        batch_size, seq_length = x.size(0), x.size(1)
        
        # Embed input
        embedded_input = self.input_embedding(x)  # Shape: (batch_size, seq_length, 10)
        
        # Process through LSTM
        lstm_output, _ = self.lstm(embedded_input)  # Shape: (batch_size, seq_length, hidden_size)
        
        all_outputs = []
        
        # For each timestep, predict states for all automata
        for t in range(seq_length):
            hidden_t = lstm_output[:, t]  # Shape: (batch_size, hidden_size)
            
            # Get predictions from each readout head
            predictions = []
            for head in self.readout_heads:
                pred = head(hidden_t)
                predictions.append(pred)
            
            # Stack predictions: (batch_size, num_automata, states_per_automaton)
            step_predictions = torch.stack(predictions, dim=1)
            all_outputs.append(step_predictions)
        
        # Stack all timesteps: (batch_size, seq_length, num_automata, states_per_automaton)
        return torch.stack(all_outputs, dim=1)