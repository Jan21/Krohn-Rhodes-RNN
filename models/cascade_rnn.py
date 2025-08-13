import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CascadeRNN(nn.Module):
    """RNN for learning cascade automaton systems with attention-based communication."""
    
    def __init__(self, input_size=1, hidden_size=100, num_automata=5, states_per_automaton=3, attention_dim=64):
        super(CascadeRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_automata = num_automata
        self.states_per_automaton = states_per_automaton
        self.attention_dim = attention_dim
        
        # Hidden state size per automaton
        self.hidden_per_automaton = hidden_size // num_automata
        if hidden_size % num_automata != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_automata ({num_automata})")
        
        # Positional encodings for each automaton ID
        self.positional_encodings = nn.Parameter(
            torch.randn(num_automata, attention_dim), requires_grad=True
        )
        
        # Key and Query projections from positional encodings
        self.key_proj = nn.Linear(attention_dim, attention_dim)
        self.query_proj = nn.Linear(attention_dim, attention_dim)
        
        # Value projection from hidden states
        self.value_proj = nn.Linear(self.hidden_per_automaton, attention_dim)
        
        # Output projection for aggregated values
        self.output_proj = nn.Linear(attention_dim, self.hidden_per_automaton)
        
        # Individual RNN cells for each automaton
        self.rnn_cells = nn.ModuleList([
            nn.RNNCell(
                input_size=input_size + self.hidden_per_automaton,  # input + aggregated values
                hidden_size=self.hidden_per_automaton,
                nonlinearity='relu'
            )
            for _ in range(num_automata)
        ])
        
        # Separate readout head for each automaton
        self.readout_heads = nn.ModuleList([
            nn.Linear(self.hidden_per_automaton, states_per_automaton) 
            for _ in range(num_automata)
        ])
    
    def plot_attention(self, attn_weights, filename):
        """
        Save the attention map as PNG.
        attn_weights: array-like (num_automata, num_automata)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # Build readable labels like A0, A1, ...
        labels = [f"A{i}" for i in range(self.num_automata)]

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            np.asarray(attn_weights),
            xticklabels=labels,
            yticklabels=labels,
            cmap="viridis",
            annot=True,
            fmt=".2f"
        )
        plt.title("Automata Attention")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        
    def compute_attention(self, hidden_states):
        """
        Compute attention-based aggregation using positional encodings.
        
        Args:
            hidden_states: (batch_size, num_automata, hidden_per_automaton)
            
        Returns:
            aggregated_values: (batch_size, num_automata, hidden_per_automaton)
        """
        
        # Generate keys and queries from positional encodings
        # Shape: (num_automata, attention_dim)
        keys = self.key_proj(self.positional_encodings)
        queries = self.query_proj(self.positional_encodings)
        
        # Generate values from hidden states
        # Shape: (batch_size, num_automata, attention_dim)
        values = self.value_proj(hidden_states)
        
        # Compute attention scores
        # queries: (num_automata, attention_dim)
        # keys: (num_automata, attention_dim)
        # attention_scores: (num_automata, num_automata)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.attention_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)

        self.attention_scores = attention_scores
        self.attention_weights = attention_weights
        
        # Apply attention to values
        # attention_weights: (num_automata, num_automata)
        # values: (batch_size, num_automata, attention_dim)
        # aggregated: (batch_size, num_automata, attention_dim)
        aggregated = torch.einsum('ij,bja->bia', attention_weights, values)
        
        # Project back to hidden dimension
        # aggregated_values: (batch_size, num_automata, hidden_per_automaton)
        aggregated_values = self.output_proj(aggregated)
        
        return aggregated_values
        
    def forward(self, x):
        batch_size, seq_length = x.size(0), x.size(1)
        
        # Initialize hidden states for all automata
        # Shape: (batch_size, num_automata, hidden_per_automaton)
        hidden_states = torch.zeros(
            batch_size, self.num_automata, self.hidden_per_automaton, 
            device=x.device
        )
        
        all_outputs = []
        
        for t in range(seq_length):
            # Current input
            current_input = x[:, t]  # Shape: (batch_size, input_size)
            
            # Compute attention-based aggregation
            aggregated_values = self.compute_attention(hidden_states)
            
            # Update each automaton's hidden state
            new_hidden_states = []
            for i in range(self.num_automata):
                # Concatenate input with aggregated value for this automaton
                rnn_input = torch.cat([
                    current_input,
                    aggregated_values[:, i]  # Shape: (batch_size, hidden_per_automaton)
                ], dim=1)
                
                # Update hidden state using the corresponding RNN cell
                new_hidden = self.rnn_cells[i](rnn_input, hidden_states[:, i])
                new_hidden_states.append(new_hidden)
            
            # Stack new hidden states
            hidden_states = torch.stack(new_hidden_states, dim=1)
            
            # Get predictions from each readout head
            predictions = []
            for i, head in enumerate(self.readout_heads):
                pred = head(hidden_states[:, i])
                predictions.append(pred)
            
            # Stack predictions: (batch_size, num_automata, states_per_automaton)
            step_predictions = torch.stack(predictions, dim=1)
            all_outputs.append(step_predictions)
        
        # Stack all timesteps: (batch_size, seq_length, num_automata, states_per_automaton)
        return torch.stack(all_outputs, dim=1)