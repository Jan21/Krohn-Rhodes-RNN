import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .rnn_model import ParityRNN


class ParityLightningModule(pl.LightningModule):
    def __init__(self, hidden_size=100, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = ParityRNN(hidden_size=hidden_size)
        self.learning_rate = learning_rate
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = F.binary_cross_entropy(y_hat, y.float())
        
        predictions = (y_hat > 0.5).float()
        accuracy = (predictions == y.float()).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', accuracy, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = F.binary_cross_entropy(y_hat, y.float())
        
        predictions = (y_hat > 0.5).float()
        accuracy = (predictions == y.float()).float().mean()
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_accuracy', accuracy, prog_bar=True, sync_dist=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        predictions = (y_hat > 0.5).float()
        accuracy = (predictions == y.float()).float().mean()
        
        self.log('test_accuracy', accuracy, sync_dist=True)
        
        return accuracy
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer