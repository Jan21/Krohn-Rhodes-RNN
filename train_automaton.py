import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data.automaton_datamodule import AutomatonDataModule
from models.automaton_lightning_module import AutomatonLightningModule

from callbacks.hidden_automaton_callback import HiddenAutomatonExtractionCallback

from data.automaton import FiniteAutomaton

# TODO: put in the config file ? 
AUTOMATON = FiniteAutomaton(num_states=2,alphabet_size=2)
AUTOMATON.accepting_states = {0}
AUTOMATON.transitions = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}

@hydra.main(version_base=None, config_path="conf", config_name="config_automaton")
def main(cfg: DictConfig) -> None:
    # Set seed
    pl.seed_everything(cfg.seed)
    
    # Initialize data module
    datamodule = AutomatonDataModule(
        automaton=AUTOMATON,
        num_states=cfg.data.num_states,
        train_seq_length=cfg.data.train_seq_length,
        test_seq_length=cfg.data.test_seq_length,
        train_samples=cfg.data.train_samples,
        val_samples=cfg.data.val_samples,
        test_samples=cfg.data.test_samples,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.seed
    )
    
    # Initialize model
    model = AutomatonLightningModule(
        hidden_size=cfg.model.hidden_size,
        learning_rate=cfg.model.learning_rate,
        num_states=cfg.model.num_states
    )
    
    # Initialize logger
    logger = WandbLogger(
        project=cfg.experiment_name,
        name=f"automaton_{cfg.data.num_states}states"
    )
    
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
        patience=10,
        mode="max"
    )

    # Extraction callback; you can also drive its params via Hydra
    extraction_cb = HiddenAutomatonExtractionCallback(
        every_n_epochs=cfg.extraction.every_n_epochs,   
        also_every_n_steps=cfg.extraction.get("every_n_steps", None),
        out_dir=cfg.extraction.out_dir,                
        eps=cfg.extraction.eps,                        
        max_len=cfg.extraction.max_len,                 
        cap_per_level=cfg.extraction.cap_per_level,    
        save_checkpoint=True,
        alphabet_symbols=datamodule.automaton.alphabet,             
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping,extraction_cb],
        accelerator="auto",
        devices="auto"
    )
    
    # Print automaton information
    print(f"Training on random automaton with {datamodule.num_states} states")
    print(f"Automaton info: {datamodule.automaton.get_info()}")
    
    # Train the model
    trainer.fit(model, datamodule)
    
    # Test the model
    trainer.test(model, datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()