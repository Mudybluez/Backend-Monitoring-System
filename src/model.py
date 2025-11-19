import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim*8),
            nn.ReLU(),
            nn.Linear(input_dim*8, input_dim*4),
            nn.ReLU(),
            nn.Linear(input_dim*4, input_dim//2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(input_dim//2, input_dim*4),
            nn.ReLU(),
            nn.Linear(input_dim*4, input_dim*8),
            nn.ReLU(),
            nn.Linear(input_dim*8, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
