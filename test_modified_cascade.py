#!/usr/bin/env python3

import torch
from models.cascade_rnn import CascadeRNN

def test_cascade_rnn():
    """Test the modified CascadeRNN model."""
    print("Testing modified CascadeRNN model...")
    
    # Model parameters
    input_size = 1
    hidden_size = 100  # Must be divisible by num_automata
    num_automata = 5
    states_per_automaton = 3
    attention_dim = 64
    
    # Create model
    model = CascadeRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_automata=num_automata,
        states_per_automaton=states_per_automaton,
        attention_dim=attention_dim
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test input
    batch_size = 32
    seq_length = 10
    x = torch.randn(batch_size, seq_length, input_size)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {seq_length}, {num_automata}, {states_per_automaton})")
    
    # Verify output shape
    expected_shape = (batch_size, seq_length, num_automata, states_per_automaton)
    assert output.shape == expected_shape, f"Output shape {output.shape} != expected {expected_shape}"
    
    print("✓ Forward pass successful!")
    
    # Test gradient computation
    x_grad = torch.randn(batch_size, seq_length, input_size, requires_grad=True)
    output_grad = model(x_grad)
    loss = output_grad.mean()
    loss.backward()
    
    print("✓ Backward pass successful!")
    
    # Test attention mechanism
    print("\nTesting attention mechanism:")
    hidden_states = torch.randn(batch_size, num_automata, hidden_size // num_automata)
    
    with torch.no_grad():
        aggregated = model.compute_attention(hidden_states)
    
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Aggregated values shape: {aggregated.shape}")
    
    expected_agg_shape = (batch_size, num_automata, hidden_size // num_automata)
    assert aggregated.shape == expected_agg_shape, f"Aggregated shape {aggregated.shape} != expected {expected_agg_shape}"
    
    print("✓ Attention mechanism working correctly!")
    
    print("\n🎉 All tests passed! The modified CascadeRNN is working correctly.")
    
    return model

if __name__ == "__main__":
    test_cascade_rnn()