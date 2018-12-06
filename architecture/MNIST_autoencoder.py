from torch import nn


class mnist_autoencoder(nn.Module):
    def __init__(self, bottleneck=3):
        super(mnist_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True), nn.Linear(128, 64), nn.ReLU(True), nn.Linear(64, bottleneck))
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True), nn.Linear(256, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x