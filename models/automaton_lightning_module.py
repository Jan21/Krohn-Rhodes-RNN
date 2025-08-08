import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .automaton_rnn import AutomatonRNN


class AutomatonLightningModule(pl.LightningModule):
    def __init__(self, hidden_size=100, learning_rate=1e-3, num_states=10):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = AutomatonRNN(hidden_size=hidden_size, output_size=num_states)
        self.learning_rate = learning_rate
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Reshape for cross entropy: (batch_size * seq_len, num_classes)
        y_hat_reshaped = y_hat.view(-1, y_hat.size(-1))
        y_reshaped = y.view(-1)
        
        loss = F.cross_entropy(y_hat_reshaped, y_reshaped)
        
        predictions = torch.argmax(y_hat, dim=-1)
        accuracy = (predictions == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', accuracy, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Reshape for cross entropy: (batch_size * seq_len, num_classes)
        y_hat_reshaped = y_hat.view(-1, y_hat.size(-1))
        y_reshaped = y.view(-1)
        
        loss = F.cross_entropy(y_hat_reshaped, y_reshaped)
        
        predictions = torch.argmax(y_hat, dim=-1)
        accuracy = (predictions == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_accuracy', accuracy, prog_bar=True, sync_dist=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        predictions = torch.argmax(y_hat, dim=-1)
        accuracy = (predictions == y).float().mean()
        
        self.log('test_accuracy', accuracy, sync_dist=True)
        
        return accuracy
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer