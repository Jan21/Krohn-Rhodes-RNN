import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from pathlib import Path

from models import ParityLightningModule
from data import ParityDataModule


def evaluate_generalization(model, test_lengths=[10, 20, 30, 40, 50, 75, 100]):
    """Evaluate model on different sequence lengths."""
    results = {}
    
    for length in test_lengths:
        print(f"\nEvaluating on sequences of length {length}...")
        
        datamodule = ParityDataModule(
            test_seq_length=length,
            test_samples=1000,
            batch_size=32,
            num_workers=4,
            seed=42
        )
        datamodule.setup("test")
        
        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )
        
        test_results = trainer.test(model, datamodule=datamodule, verbose=False)
        accuracy = test_results[0]['test_accuracy']
        
        results[length] = accuracy
        print(f"Length {length}: Accuracy = {accuracy:.4f}")
    
    return results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint_dir = Path("lightning_logs")
    
    if not checkpoint_dir.exists():
        print("No checkpoints found. Please run training first.")
        return
    
    checkpoint_path = None
    for version_dir in checkpoint_dir.iterdir():
        if version_dir.is_dir() and version_dir.name.startswith("version_"):
            checkpoints_dir = version_dir / "checkpoints"
            if checkpoints_dir.exists():
                for ckpt in checkpoints_dir.glob("*.ckpt"):
                    checkpoint_path = ckpt
                    break
                if checkpoint_path:
                    break
    
    if checkpoint_path is None:
        print("No checkpoint found. Please run training first.")
        return
    
    print(f"Loading model from: {checkpoint_path}")
    model = ParityLightningModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    print("Evaluating generalization to different sequence lengths...")
    results = evaluate_generalization(model)
    
    print("\n" + "="*50)
    print("GENERALIZATION RESULTS")
    print("="*50)
    
    for length, accuracy in results.items():
        if length == 10:
            print(f"Length {length:3d} (training): {accuracy:.4f}")
        else:
            print(f"Length {length:3d}:            {accuracy:.4f}")
    
    print("\nGeneralization analysis:")
    train_acc = results[10]
    long_accs = [results[length] for length in [30, 50, 75, 100]]
    avg_long_acc = sum(long_accs) / len(long_accs)
    
    print(f"Training length (10) accuracy: {train_acc:.4f}")
    print(f"Average long sequence accuracy: {avg_long_acc:.4f}")
    print(f"Generalization gap: {train_acc - avg_long_acc:.4f}")


if __name__ == "__main__":
    main()