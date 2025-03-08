# model.py
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, window_size):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        
        self.gru1 = nn.GRU(input_dim, hidden_dim, num_layers=1, 
                           batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(2 * hidden_dim, hidden_dim, num_layers=1, 
                           batch_first=True, bidirectional=True)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2 * window_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), 
        )

    def forward(self, x):
        batch_size = x.size(0)
        x, _ = self.gru1(x)
        x = nn.functional.layer_norm(x, x.size()[-1:])
        x, _ = self.gru2(x)
        x = nn.functional.layer_norm(x, x.size()[-1:]) 
        output = self.readout(x.reshape(batch_size, -1))
        return output
