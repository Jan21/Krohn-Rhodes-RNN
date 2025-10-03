import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .cascade_rnn import CascadeRNN


class CascadeLightningModule(pl.LightningModule):
    def __init__(self, attention_type = None, hidden_size=100, learning_rate=1e-3, num_automata=5, states_per_automaton=3):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = CascadeRNN(
            hidden_size=hidden_size,
            num_automata=num_automata,
            states_per_automaton=states_per_automaton,
            attention=attention_type
        )
        self.learning_rate = learning_rate
        self.num_automata = num_automata
        self.states_per_automaton = states_per_automaton
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  # Shape: (batch_size, seq_length, num_automata, states_per_automaton)
        
        # Compute loss for each automaton separately
        total_loss = 0
        total_accuracy = 0
        
        for automaton_idx in range(self.num_automata):
            # Get predictions and targets for this automaton
            y_hat_auto = y_hat[:, :, automaton_idx, :]  # (batch_size, seq_length, states_per_automaton)
            y_auto = y[:, :, automaton_idx]  # (batch_size, seq_length)
            
            # Reshape for cross entropy
            y_hat_reshaped = y_hat_auto.reshape(-1, self.states_per_automaton)
            y_reshaped = y_auto.reshape(-1)
            
            # Compute loss and accuracy for this automaton
            loss = F.cross_entropy(y_hat_reshaped, y_reshaped)
            predictions = torch.argmax(y_hat_auto, dim=-1)
            accuracy = (predictions == y_auto).float().mean()
            
            total_loss += loss
            total_accuracy += accuracy
            
            # Log per-automaton metrics
            self.log(f'train_loss_auto_{automaton_idx}', loss, prog_bar=False)
            self.log(f'train_acc_auto_{automaton_idx}', accuracy, prog_bar=False)
        
        # Average across automata
        avg_loss = total_loss / self.num_automata
        avg_accuracy = total_accuracy / self.num_automata
        
        self.log('train_loss', avg_loss, prog_bar=True)
        self.log('train_accuracy', avg_accuracy, prog_bar=True)
        
        return avg_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        total_loss = 0
        total_accuracy = 0
        
        for automaton_idx in range(self.num_automata):
            y_hat_auto = y_hat[:, :, automaton_idx, :]
            y_auto = y[:, :, automaton_idx]
            
            y_hat_reshaped = y_hat_auto.reshape(-1, self.states_per_automaton)
            y_reshaped = y_auto.reshape(-1)
            
            loss = F.cross_entropy(y_hat_reshaped, y_reshaped)
            predictions = torch.argmax(y_hat_auto, dim=-1)
            accuracy = (predictions == y_auto).float().mean()
            
            total_loss += loss
            total_accuracy += accuracy
            
            self.log(f'val_loss_auto_{automaton_idx}', loss, prog_bar=False, sync_dist=True)
            self.log(f'val_acc_auto_{automaton_idx}', accuracy, prog_bar=False, sync_dist=True)
        
        avg_loss = total_loss / self.num_automata
        avg_accuracy = total_accuracy / self.num_automata
        
        self.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
        self.log('val_accuracy', avg_accuracy, prog_bar=True, sync_dist=True)
        
        return avg_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        total_accuracy = 0
        
        for automaton_idx in range(self.num_automata):
            y_hat_auto = y_hat[:, :, automaton_idx, :]
            y_auto = y[:, :, automaton_idx]
            
            predictions = torch.argmax(y_hat_auto, dim=-1)
            accuracy = (predictions == y_auto).float().mean()
            
            total_accuracy += accuracy
            
            self.log(f'test_acc_auto_{automaton_idx}', accuracy, sync_dist=True)
        
        avg_accuracy = total_accuracy / self.num_automata
        self.log('test_accuracy', avg_accuracy, sync_dist=True)
        
        return avg_accuracy
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def get_attention_weights(self):
        """
        Return the last attention matrix (num_automata x num_automata) as a CPU numpy array.
        Requires that a forward pass has happened (training/validation just ran).
        """
        W = getattr(self.model, "attention_weights", None)
        if W is None:
            raise RuntimeError("No attention weights cached yet. Has forward() run?")
        return W.detach().cpu().numpy()