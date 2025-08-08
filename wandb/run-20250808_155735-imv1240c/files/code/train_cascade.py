#!/usr/bin/env python3

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate



@hydra.main(version_base=None, config_path="conf", config_name="config_cascade")
def main(cfg: DictConfig):
    # Set seed
    pl.seed_everything(cfg.seed)
    
    # Initialize wandb logger
    wandb_logger = pl.loggers.WandbLogger(
        project="cascade-rnn",
        name=cfg.experiment_name,
        config=dict(cfg)
    )
    
    # Initialize data module
    datamodule = instantiate(cfg.data)
    
    # Initialize model
    model = instantiate(cfg.model)
    
    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
        filename="best-checkpoint-{epoch:02d}-{val_accuracy:.3f}",
        save_weights_only=False
    )
    
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        mode="max"
    )
    
    # Initialize trainer
    trainer = instantiate(cfg.trainer, callbacks=[checkpoint_callback, early_stopping], logger=wandb_logger)
    
    # Print and log cascade system information
    system_info = {
        "num_automata": datamodule.num_automata,
        "states_per_automaton": datamodule.states_per_automaton,
        "alphabet_size": datamodule.alphabet_size,
        "train_seq_length": datamodule.train_seq_length,
        "test_seq_length": datamodule.test_seq_length
    }
    
    print(f"Training on cascade system:")
    print(f"  - {system_info['num_automata']} automata")
    print(f"  - {system_info['states_per_automaton']} states per automaton") 
    print(f"  - Alphabet size: {system_info['alphabet_size']}")
    print(f"  - Training on sequences of length {system_info['train_seq_length']}")
    print(f"  - Testing on sequences of length {system_info['test_seq_length']}")
    
    cascade_info = datamodule.cascade_system.get_info()
    print(f"\nCascade system info: {cascade_info}")
    
    # Log system configuration to wandb
    wandb_logger.experiment.config.update(system_info)
    
    # Convert cascade_info to a serializable format
    serializable_cascade_info = {}
    if isinstance(cascade_info, dict):
        for key, value in cascade_info.items():
            # Convert tuple keys to strings
            str_key = str(key) if not isinstance(key, (str, int, float, bool)) else key
            # Convert complex values to strings if needed
            if isinstance(value, (list, tuple, dict)) and not all(isinstance(v, (str, int, float, bool, type(None))) for v in (value if isinstance(value, (list, tuple)) else value.values() if isinstance(value, dict) else [value])):
                serializable_cascade_info[str_key] = str(value)
            else:
                serializable_cascade_info[str_key] = value
    else:
        serializable_cascade_info = {"info": str(cascade_info)}
    
    wandb_logger.experiment.config.update({"cascade_system_info": serializable_cascade_info})
    
    # Train the model
    trainer.fit(model, datamodule)
    # Test the model for length generalization
    print(f"\n{'='*60}")
    print("Testing length generalization on cascade system...")
    print(f"{'='*60}")
    test_results = trainer.test(model, datamodule, ckpt_path="best")
    
    print(f"\nLength Generalization Results:")
    print(f"Train length: {datamodule.train_seq_length}")
    print(f"Test length: {datamodule.test_seq_length}")
    print(f"Average test accuracy: {test_results[0]['test_accuracy']:.4f}")
    
    # Print per-automaton results
    for i in range(datamodule.num_automata):
        acc_key = f'test_acc_auto_{i}'
        if acc_key in test_results[0]:
            print(f"Automaton {i} test accuracy: {test_results[0][acc_key]:.4f}")


if __name__ == "__main__":
    main()