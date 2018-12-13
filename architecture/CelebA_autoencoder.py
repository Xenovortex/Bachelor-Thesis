from torch import nn


class celeba_autoencoder(nn.Module):

    def __init__(self, bottleneck=500): # input: 3, 218, 178 (original: 3, 218, 178)
        super(celeba_autoencoder, self).__init__()

        # encoder
        self.e1 = nn.Conv2d(3, 32, 4, stride=1, padding=0)  # b, 32, 215, 175
        self.e2 = nn.Conv2d(32, 64, 4, stride=1, padding=0) # b, 64, 212, 172
        self.e3 = nn.MaxPool2d(2, stride=2, padding=0) # b, 64, 106, 86
        self.e4 = nn.Conv2d(64, 128, 4, stride=1, padding=0) # b, 128, 103, 83
        self.e5 = nn.Conv2d(128, 256, 4, stride=1, padding=0) # b, 256, 100, 80
        self.e6 = nn.MaxPool2d(2, stride=2, padding=0) # b, 256, 50, 40
        self.e7 = nn.Conv2d(256, 512, 4, stride=1, padding=0) # b, 512, 47, 37
        self.e8 = nn.Conv2d(512, 1024, 4, stride=1, padding=0) # b, 1024, 44, 34
        self.e9 = nn.MaxPool2d(2, stride=2, padding=0) # b, 1024, 22, 17
        self.e10 = nn.Conv2d(1024, 1024, 3, stride=1, padding=0) # b, 1024, 20, 15
        self.e11 = nn.Linear(1024*20*15, bottleneck) # bottleneck

        # decoder
        self.d1 = nn.Linear(bottleneck, 1024*20*15)  # b, 1024, 20, 15
        self.d2 = nn.ConvTranspose2d(1024, 1024, 3, stride=1, padding=0) # b, 1024, 22, 17
        self.d3 = nn.Upsample(scale_factor=2) # b, 1024, 44, 34
        self.d4 = nn.ConvTranspose2d(1024, 512, 4, stride=1, padding=0) # b, 512, 47, 37
        self.d5 = nn.ConvTranspose2d(512, 256, 4, stride=1, padding=0) # b, 256, 50, 40
        self.d6 = nn.Upsample(scale_factor=2) # b, 256, 100, 80
        self.d7 = nn.ConvTranspose2d(256, 128, 4, stride=1, padding=0) # b, 128, 103, 83
        self.d8 = nn.ConvTranspose2d(128, 64, 4, stride=1, padding=0) # b, 64, 106, 86
        self.d9 = nn.Upsample(scale_factor=2) # b, 64, 212, 172
        self.d10 = nn.ConvTranspose2d(64, 32, 4, stride=1, padding=0) # b, 32, 215, 175
        self.d11 = nn.ConvTranspose2d(32, 3, 4, stride=1, padding=0) # b, 3, 218, 178

        # activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def encode(self, x):
        h1 = self.relu(self.e1(x))
        h2 = self.relu(self.e2(h1))
        h3 = self.relu(self.e4(self.e3(h2)))
        h4 = self.relu(self.e5(h3))
        h5 = self.relu(self.e7(self.e6(h4)))
        h6 = self.relu(self.e8(h5))
        h7 = self.relu(self.e10(self.e9(h6)))
        h7 = h7.view(-1, 1024*20*15)
        h8 = self.e11(h7)

        return h8


    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, 1024, 20, 15)
        h2 = self.relu(self.d2(h1))
        h3 = self.relu(self.d4(self.d3(h2)))
        h4 = self.relu(self.d5(h3))
        h5 = self.relu(self.d7(self.d6(h4)))
        h6 = self.relu(self.d8(h5))
        h7 = self.relu(self.d10(self.d9(h6)))
        h8 = self.tanh(self.d11(h7))

        return h8


    def forward(self, x):
        z = self.encode(x)
        x_ = self.decode(z)

        return x_
