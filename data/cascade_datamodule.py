# datamodule.py
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .cascade_dataset import CascadeDataset
from .cascade_automaton import generate_cascade_system               # old (state+prev-state dependent)
from .output_cascade_automaton import generate_output_cascade_system # new (Mealy-style)
from .adapters import OutputCascadeAdapter

class CascadeDataModule(pl.LightningDataModule):
    def __init__(self,
        num_automata=5,
        states_per_automaton=3,
        alphabet_size=2,
        train_seq_length=10,
        test_seq_length=50,
        train_samples=1000,
        val_samples=2000,
        test_samples=1000,
        batch_size=32,
        num_workers=4,
        seed=42,
        use_output_cascade: bool = False,   # <— NEW
    ):
        super().__init__()
        self.num_automata = num_automata
        self.states_per_automaton = states_per_automaton
        self.alphabet_size = alphabet_size
        self.train_seq_length = train_seq_length
        self.test_seq_length = test_seq_length
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.use_output_cascade = use_output_cascade

        self.save_hyperparameters()

        # Generate the cascade system once
        if self.use_output_cascade:
            base = generate_output_cascade_system(
                num_automata=num_automata,
                states_per_automaton=states_per_automaton,
                alphabet_size=alphabet_size,
                seed=seed,
            )
            self.cascade_system = OutputCascadeAdapter(base)
        else:
            self.cascade_system = generate_cascade_system(
                num_automata=num_automata,
                states_per_automaton=states_per_automaton,
                alphabet_size=alphabet_size,
                seed=seed,
            )
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = CascadeDataset(
                cascade_system=self.cascade_system,
                num_samples=self.train_samples,
                seq_length=self.train_seq_length,
                seed=self.seed
            )
            
            self.val_dataset = CascadeDataset(
                cascade_system=self.cascade_system,
                num_samples=self.val_samples,
                seq_length=self.test_seq_length,
                seed=self.seed + 1
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = CascadeDataset(
                cascade_system=self.cascade_system,
                num_samples=self.test_samples,
                seq_length=self.test_seq_length,
                seed=self.seed + 2
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )