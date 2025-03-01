import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, unpack_sequence, pack_sequence

class LSTM(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers, num_classes, static_input=False, static_input_dim=None, use_all_seq=False):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.static_input = static_input
        self.static_input_dim = static_input_dim
        self.use_all_seq = use_all_seq

        self.input_mapping = nn.Linear(input_dim, latent_dim)
        self.lstm = nn.LSTM(latent_dim, latent_dim, num_layers, batch_first=True)
        if static_input:
            self.fc = nn.Linear(latent_dim + static_input_dim, num_classes)
        else:
            self.fc = nn.Linear(latent_dim, num_classes)
    
    def forward(self, x):
        if self.static_input:
            x, static_input = x
        
        if isinstance(x, PackedSequence):
            x_ = unpack_sequence(x)
            x_ = [self.input_mapping(x_i) for x_i in x_]
            x = pack_sequence(x_, enforce_sorted=False)
        else:
            x = self.input_mapping(x)

        out, _ = self.lstm(x)

        if isinstance(out, PackedSequence):
            out = unpack_sequence(out)
            if self.use_all_seq:
                if self.static_input:
                    out = [torch.cat([x_i, static_input_i.unsqueeze(0).expand(len(x_i), -1)], dim=1) 
                        for x_i, static_input_i in zip(out, static_input)]
                out = [self.fc(x_i) for x_i in out]
                out = pack_sequence(out, enforce_sorted=False)
            else:
                out = torch.stack([x_i[-1, :] for x_i in out])
                if self.static_input:
                    out = torch.cat([out, static_input], dim=1)
                out = self.fc(out)
        else:
            out = out[:, -1, :]
            if self.static_input:
                out = torch.cat([out, static_input], dim=1)
            out = self.fc(out)
        return out