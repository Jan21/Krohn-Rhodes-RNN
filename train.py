import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch

from models import ParityLightningModule
from data import ParityDataModule


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    logger = hydra.utils.instantiate(cfg.logger, name=cfg.experiment_name)
    
    callbacks = [
        ModelCheckpoint(
            monitor="val_accuracy",
            mode="max",
            save_top_k=1,
            filename="best-checkpoint-{epoch:02d}-{val_accuracy:.3f}",
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=10,
            verbose=True,
        ),
    ]
    
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )
    
    trainer.fit(model, datamodule=datamodule)
    
    print("\nTraining completed! Now testing on longer sequences...")
    test_results = trainer.test(model, datamodule=datamodule)
    
    print(f"\nTest Results (Length {datamodule.test_seq_length}):")
    print(f"Test Accuracy: {test_results[0]['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()