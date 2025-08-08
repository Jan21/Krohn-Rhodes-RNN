#!/usr/bin/env python3

from data.automaton import generate_random_automaton
from data.automaton_dataset import AutomatonDataset
import torch

def test_automaton():
    """Test the random automaton generation and dataset creation."""
    
    # Generate a random 10-state automaton
    print("Generating random automaton with 10 states...")
    automaton = generate_random_automaton(num_states=10, seed=42)
    
    print(f"Automaton info: {automaton.get_info()}")
    
    # Test on a simple sequence
    test_sequence = [0, 1, 1, 0, 1]
    states, accepts = automaton.run(test_sequence)
    
    print(f"\nTest sequence: {test_sequence}")
    print(f"States visited: {states}")
    print(f"Accepts at each step: {accepts}")
    print(f"Final acceptance: {automaton.accepts_sequence(test_sequence)}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = AutomatonDataset(automaton, num_samples=100, seq_length=5, seed=42)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Show first few samples
    for i in range(3):
        seq, labels = dataset[i]
        print(f"Sample {i}: sequence={seq.squeeze().tolist()}, labels={labels.squeeze().tolist()}")
    
    print("\nAutomaton test completed successfully!")

if __name__ == "__main__":
    test_automaton()