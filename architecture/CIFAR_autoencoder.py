from torch import nn

class cifar_autoencoder(nn.Module):
    def __init__(self, bottleneck=32):
        super(cifar_autoencoder, self).__init__()

        # encoder
        self.e1 = nn.Conv2d(3, 128, 4, stride=2, padding=1)  # b, 128, 16, 16)
        self.e2 = nn.MaxPool2d(2, stride=2)  # b, 128, 8, 8
        self.e3 = nn.Conv2d(128, 256, 2, stride=2, padding=1)  # b, 256, 5, 5
        self.e4 = nn.MaxPool2d(2, stride=1)  # b, 256, 4, 4
        self.e5 = nn.Conv2d(256, 512, 4, stride=1, padding=1)  # b, 512, 3, 3
        self.e6 = nn.Linear(512*3*3, bottleneck)  # bottleneck

        # decoder
        self.d1 = nn.Linear(bottleneck, 512*3*3)  # b, 512, 3, 3
        self.d2 = nn.ConvTranspose2d(512, 128, 3, stride=2)  # b, 128, 7, 7
        self.d3 = nn.ConvTranspose2d(128, 32, 4, stride=2)  # b, 32, 16, 16
        self.d4 = nn.ConvTranspose2d(32, 3, 2, stride=2)  # b, 3, 32, 32

        # activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def encode(self, x):
        h1 = self.relu(self.e1(x))
        h2 = self.relu(self.e3(self.e2(h1)))
        h3 = self.relu(self.e5(self.e4(h2)))
        h3 = h3.view(-1, 512*3*3)
        h4 = self.e6(h3)

        return  h4


    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, 512, 3, 3)
        h2 = self.relu(self.d2(h1))
        h3 = self.relu(self.d3(h2))
        h4 = self.tanh(self.d4(h3))

        return h4


    def forward(self, x):
        z = self.encode(x)
        x_ = self.decode(z)

        return x_
