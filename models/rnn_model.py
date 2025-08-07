import torch
import torch.nn as nn


class ParityRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1):
        super(ParityRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            nonlinearity='relu'
        )
        
        self.readout = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        
        rnn_out, _ = self.rnn(x, h0)
        
        predictions = self.readout(rnn_out)
        
        return torch.sigmoid(predictions)