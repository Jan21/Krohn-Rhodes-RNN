import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CascadeRNN(nn.Module):
    """LSTM-based model for learning cascade automaton systems with attention-based communication."""
    
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
        
        # Shared input embedding layer for all automata
        self.input_embedding = nn.Linear(input_size, 10)
        
        # Separate pre-state embeddings for each automaton to avoid gradient interference
        self.pre_state_embeddings = nn.ModuleList([
            nn.Embedding(states_per_automaton, 30) for _ in range(num_automata)
        ])
        
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
        
        # Individual LSTM cells for each automaton
        self.rnn_cells = nn.ModuleList([
            nn.LSTMCell(
                input_size=40,#10 + self.hidden_per_automaton,  # embedded input + aggregated values
                hidden_size=self.hidden_per_automaton
            )
            for _ in range(num_automata)
        ])
        
        # Separate readout head for each automaton
        self.readout_heads = nn.ModuleList([
            nn.Linear(self.hidden_per_automaton, states_per_automaton) 
            for _ in range(num_automata)
        ])
        
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
        # attention_scores = attention_scores / math.sqrt(self.attention_dim)
        # attention_weights = F.softmax(attention_scores, dim=-1)
        # Create fixed attention pattern where each automaton only attends to previous one
        # Shape: (num_automata, num_automata)
        fixed_attention = torch.zeros_like(attention_scores)
        
        # Set attention to previous automaton (except first one)
        indices = torch.arange(1, self.num_automata)
        fixed_attention[indices, indices-1] = 1.0
        
        # Override the computed attention weights with fixed pattern
        attention_weights = fixed_attention
        
        # Apply attention to values
        # attention_weights: (num_automata, num_automata)
        # values: (batch_size, num_automata, attention_dim)
        # aggregated: (batch_size, num_automata, attention_dim)
        aggregated = torch.einsum('ij,bja->bia', attention_weights, values)
        
        # Project back to hidden dimension
        # aggregated_values: (batch_size, num_automata, hidden_per_automaton)
        aggregated_values = self.output_proj(aggregated)
        
        return aggregated_values
        
    def forward(self, x, y, train):
        batch_size, seq_length = x.size(0), x.size(1)
        
        # Initialize hidden states and cell states for all automata
        # Shape: (batch_size, num_automata, hidden_per_automaton)
        hidden_states = torch.zeros(
            batch_size, self.num_automata, self.hidden_per_automaton, 
            device=x.device
        )
        cell_states = torch.zeros(
            batch_size, self.num_automata, self.hidden_per_automaton, 
            device=x.device
        )
        all_outputs = []
        
        for t in range(seq_length):
            # Current input (will be embedded separately for each automaton)
            current_input = x[:, t]  # Shape: (batch_size, input_size)
            
            # Process automata sequentially within this timestep
            step_predictions = []
            new_hidden_states = []
            new_cell_states = []
            #current_automata_states = []  # Store predicted states for this timestep
            
            if train:
                # PARALLEL PROCESSING DURING TRAINING (with stop-gradient)
                # Process all automata in parallel using ground truth dependencies
                # This avoids gradient flow through the cascade dependencies
                
                for i in range(self.num_automata):
                    # Shared embedded input across all automata
                    embedded_input = self.input_embedding(current_input)  # Shape: (batch_size, 10)
                    
                    if i == 0:
                        # First automaton is independent
                        if t == 0:
                            embedded_dependency = torch.zeros(batch_size, 30).to(x.device)
                        else:
                            prev_state = y[:, t-1, i]
                            embedded_dependency = self.pre_state_embeddings[i](prev_state)
                    else:
                        # Use ground truth dependency with detached state
                        # This prevents gradient flow through the dependency chain
                        prev_automaton_state = y[:, t-1, i-1].detach()  # Detach to stop gradients
                        embedded_dependency = self.pre_state_embeddings[i](prev_automaton_state)
                    
                    # Concatenate embedded input with dependency information
                    lstm_input = torch.cat([
                        embedded_input,        # Shape: (batch_size, 10) - NOW SEPARATE PER AUTOMATON
                        embedded_dependency    # Shape: (batch_size, 30)
                    ], dim=1)  # Shape: (batch_size, 40)
                    
                    # Update hidden state using the corresponding LSTM cell
                    new_hidden, new_cell = self.rnn_cells[i](lstm_input, (hidden_states[:, i], cell_states[:, i]))
                    new_hidden_states.append(new_hidden)
                    new_cell_states.append(new_cell)
                    
                    # Get prediction from readout head
                    pred = self.readout_heads[i](new_hidden)  # Shape: (batch_size, states_per_automaton)
                    step_predictions.append(pred)
            
            else:
                # SEQUENTIAL PROCESSING DURING INFERENCE
                # Process automata sequentially using predictions from previous automata
                
                for i in range(self.num_automata):
                    # Shared embedded input across all automata
                    embedded_input = self.input_embedding(current_input)  # Shape: (batch_size, 10)
                    
                    if i == 0:
                        # First automaton is independent
                        if t == 0:
                            embedded_dependency = torch.zeros(batch_size, 30).to(x.device)
                        else:
                            prev_state = y[:, t-1, i]
                            embedded_dependency = self.pre_state_embeddings[i](prev_state)
                    else:
                        # Use prediction from previous automaton at previous timestep
                        if t == 0:
                            # At first timestep, use initial state (0)
                            prev_automaton_state = torch.zeros(batch_size, dtype=torch.long).to(x.device)
                        else:
                            # Use prediction from previous timestep for previous automaton
                            prev_automaton_logits = all_outputs[t-1][:, i-1]  # Shape: (batch_size, states_per_automaton)
                            prev_automaton_state = torch.argmax(prev_automaton_logits, dim=-1)  # Shape: (batch_size,)
                        
                        embedded_dependency = self.pre_state_embeddings[i](prev_automaton_state)
                    
                    # Concatenate embedded input with dependency information
                    lstm_input = torch.cat([
                        embedded_input,        # Shape: (batch_size, 10) - NOW SEPARATE PER AUTOMATON
                        embedded_dependency    # Shape: (batch_size, 30)
                    ], dim=1)  # Shape: (batch_size, 40)
                    
                    # Update hidden state using the corresponding LSTM cell
                    new_hidden, new_cell = self.rnn_cells[i](lstm_input, (hidden_states[:, i], cell_states[:, i]))
                    new_hidden_states.append(new_hidden)
                    new_cell_states.append(new_cell)
                    
                    # Get prediction from readout head
                    pred = self.readout_heads[i](new_hidden)  # Shape: (batch_size, states_per_automaton)
                    step_predictions.append(pred)
            
            # Stack new hidden states and cell states
            hidden_states = torch.stack(new_hidden_states, dim=1)
            cell_states = torch.stack(new_cell_states, dim=1)
            
            # Stack predictions: (batch_size, num_automata, states_per_automaton)
            step_predictions_tensor = torch.stack(step_predictions, dim=1)
            all_outputs.append(step_predictions_tensor)
        
        # Stack all timesteps: (batch_size, seq_length, num_automata, states_per_automaton)
        return torch.stack(all_outputs, dim=1)