#!/usr/bin/env python3

from data.cascade_automaton import generate_cascade_system
from data.cascade_dataset import CascadeDataset
import torch

def test_cascade():
    """Test the cascade automaton generation and dataset creation."""
    
    # Generate a cascade system: 5 automata with 3 states each
    print("Generating cascade system with 5 automata, 3 states each...")
    cascade_system = generate_cascade_system(
        num_automata=5, 
        states_per_automaton=3, 
        alphabet_size=2, 
        seed=42
    )
    
    print(f"Cascade system info: {cascade_system.get_info()}")
    
    # Test on a simple sequence
    test_sequence = [0, 1, 1, 0, 1]
    all_states = cascade_system.run(test_sequence)
    
    print(f"\nTest sequence: {test_sequence}")
    for i, states in enumerate(all_states):
        print(f"Automaton {i} states: {states}")
    
    # Create dataset
    print("\nCreating cascade dataset...")
    dataset = CascadeDataset(cascade_system, num_samples=100, seq_length=5, seed=42)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Input shape: {dataset[0][0].shape}")  # Should be (seq_length, 1)
    print(f"Label shape: {dataset[0][1].shape}")  # Should be (seq_length, num_automata)
    
    # Show first few samples
    for i in range(3):
        seq, labels = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Sequence: {seq.squeeze().tolist()}")
        print(f"  States shape: {labels.shape}")
        for j in range(cascade_system.num_automata):
            print(f"  Automaton {j} states: {labels[:, j].tolist()}")
    
    print("\nCascade system test completed successfully!")

if __name__ == "__main__":
    test_cascade()