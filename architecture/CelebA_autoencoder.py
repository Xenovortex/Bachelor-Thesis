from torch import nn


class celeba_autoencoder(nn.Module):

    def __init__(self, bottleneck=500): # input: 3, 218, 178 (original: 3, 218, 178)
        super(celeba_autoencoder, self).__init__()

        # encoder
        self.e1 = nn.Conv2d(3, 128, 4, stride=2, padding=0)  # b, 128, 108, 88
        self.e2 = nn.MaxPool2d(2, stride=2, padding=1)  # b, 128, 54, 44
        self.e3 = nn.Conv2d(128, 128*2, 4, stride=2, padding=(0,1))  # b, 256, 26, 22
        self.e4 = nn.Conv2d(128*2, 128*4, 4, stride=2, padding=0)  # b, 512, 12, 10
        self.e5 = nn.Conv2d(128*4, 128*8, 4, stride=2, padding=0)  # b, 1024, 5, 4
        self.e6 = nn.Conv2d(128*8, 128*8, 4, stride=1, padding=1)  # b, 1024, 4, 3
        self.e7 = nn.Linear(128*8*4*3, bottleneck)  # bottleneck

        # decoder
        self.d1 = nn.Linear(bottleneck, 128*8*4*3)  # b, 1024, 4, 3
        self.d2 = nn.ConvTranspose2d(128*8, 512, 5, stride=2)  # b, 512, 11, 9
        self.d3 = nn.ConvTranspose2d(512, 256, 5, stride=2)  # b, 256, 25, 21
        self.d4 = nn.ConvTranspose2d(256, 128, 5, stride=2, padding=(0,1))  # b, 128, 53, 43
        self.d5 = nn.ConvTranspose2d(128, 64, 4, stride=2)  # b, 64, 108, 88
        self.d6 = nn.ConvTranspose2d(64, 3, 4, stride=2)  # b, 3, 218, 178

        # activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def encode(self, x):
        h1 = self.relu(self.e1(x))
        h2 = self.relu(self.e3(self.e2(h1)))
        h3 = self.relu(self.e4(h2))
        h4 = self.relu(self.e5(h3))
        h5 = self.relu(self.e6(h4))
        h5 = h5.view(-1, 128*8*4*3)
        h6 = self.e7(h5)

        return h6


    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, 128*8, 4, 3)
        h2 = self.relu(self.d2(h1))
        h3 = self.relu(self.d3(h2))
        h4 = self.relu(self.d4(h3))
        h5 = self.relu(self.d5(h4))
        h6 = self.tanh(self.d6(h5))

        return h6

    def forward(self, x):
        z = self.encode(x)
        x_ = self.decode(z)

        return x_
