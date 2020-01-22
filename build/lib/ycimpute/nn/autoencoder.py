
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, dim,theta):
        super(Autoencoder, self).__init__()
        self.dim = dim

        self.drop_out = nn.Dropout(p=0.1)

        self.encoder = nn.Sequential(
            nn.Linear(dim + theta * 0, dim + theta * 1),
            nn.Tanh(),
            nn.Linear(dim + theta * 1, dim + theta * 2),
            nn.Tanh(),
            nn.Linear(dim + theta * 2, dim + theta * 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim + theta * 3, dim + theta * 2),
            nn.Tanh(),
            nn.Linear(dim + theta * 2, dim + theta * 1),
            nn.Tanh(),
            nn.Linear(dim + theta * 1, dim + theta * 0)
        )

    def forward(self, x):
        x = x.view(-1, self.dim)
        x_missed = self.drop_out(x)

        z = self.encoder(x_missed)
        out = self.decoder(z)

        out = out.view(-1, self.dim)

        return out