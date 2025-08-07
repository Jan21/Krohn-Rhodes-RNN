import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .parity_dataset import ParityDataset


class ParityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_seq_length=10,
        test_seq_length=50,
        train_samples=1000,
        val_samples=2000,
        test_samples=1000,
        batch_size=32,
        num_workers=4,
        seed=42
    ):
        super().__init__()
        self.train_seq_length = train_seq_length
        self.test_seq_length = test_seq_length
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        self.save_hyperparameters()
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = ParityDataset(
                num_samples=self.train_samples,
                seq_length=self.train_seq_length,
                seed=self.seed
            )
            
            self.val_dataset = ParityDataset(
                num_samples=self.val_samples,
                seq_length=self.test_seq_length,
                seed=self.seed + 1
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = ParityDataset(
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