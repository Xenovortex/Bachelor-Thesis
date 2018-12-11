from torch import nn


class celeba_autoencoder(nn.Module):

    def __init__(self, bottleneck=500): # input: 3, 218, 178 (original: 3, 218, 178)
        super(celeba_autoencoder, self).__init__()

        # encoder
        self.e1 = nn.Conv2d(3, 32, 3, stride=1, padding=0)  # b, 32, 216, 176
        self.e2 = nn.Conv2d(32, 64, 3, stride=1, padding=0) # b, 64, 214, 174
        self.e3 = nn.Conv2d(64, 128, 3, stride=1, padding=0) # b, 128, 212, 172
        self.e4 = nn.MaxPool2d(2, stride=2, padding=0) # b, 128, 106, 86
        self.e5 = nn.Conv2d(128, 256, 3, stride=1, padding=0) # b, 256, 104, 84
        self.e6 = nn.Conv2d(256, 512, 3, stride=1, padding=0) # b, 512, 102, 82
        self.e7 = nn.Conv2d(512, 1024, 3, stride=1, padding=0) # b, 1024, 100, 80
        self.e8 = nn.MaxPool2d(2, stride=2, padding=0) # b, 1024, 50, 40
        self.e9 = nn.Conv2d(1024, 1024, 3, stride=1, padding=0) # b, 1024, 48, 38
        self.e10 = nn.Conv2d(1024, 1024, 3, stride=1, padding=0) # b, 1024, 46, 36
        self.e11 = nn.Linear(1024*46*36, bottleneck) # bottleneck

        # decoder
        self.d1 = nn.Linear(bottleneck, 1024*46*36)  # b, 1024, 46, 36
        self.d2 = nn.ConvTranspose2d(1024, 1024, 3, stride=1, padding=0) # 1024, 48, 38
        self.d3 = nn.ConvTranspose2d(1024, 1024, 3, stride=1, padding=0) # 1024, 50, 40
        self.d4 = nn.UpsamlingNearest2d(scale_factor=2) # 1024, 100, 80
        self.d5 = nn.ConvTranspose2d(1024, 512, 3, stride=1, padding=0) # 512, 102, 82
        self.d6 = nn.ConvTranspose2d(512, 256, 3, stride=1, padding=0) # 256, 104, 84
        self.d7 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=0) # 128, 106, 86
        self.d8 = nn.UpsamlingNearest2d(scale_factor=2) # 128, 212, 172
        self.d9 = nn.ConvTranspose2d(128, 64, 3, stride=1, padding=0) # 64, 214, 174
        self.d10 = nn.ConvTranspose2d(64, 32, 3, stride=1, padding=0) # 32, 216, 176
        self.d11 = nn.ConvTranspose2d(32, 3, 3, stride=1, padding=0) # 3, 218, 178

        # activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def encode(self, x):
        h1 = self.relu(self.e1(x))
        h2 = self.relu(self.e2(h1))
        h3 = self.relu(self.e4(self.e3(h2)))
        h4 = self.relu(self.e5(h3))
        h5 = self.relu(self.e6(h4))
        h6 = self.relu(self.e8(self.e7(h5)))
        h7 = self.relu(self.e9(h6))
        h8 = self.relu(self.e10(h7))
        h8 = h8.view(-1, 1024*46*36)
        h9 = self.e11(h8)

        return h9


    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, 1024, 46, 36)
        h2 = self.relu(self.d2(h1))
        h3 = self.relu(self.d3(h2))
        h4 = self.relu(self.d5(self.d4(h3)))
        h5 = self.relu(self.d6(h4))
        h6 = self.relu(self.d7(h5))
        h7 = self.relu(self.d9(self.d8(h6)))
        h8 = self.relu(self.d10(h7))
        h9 = self.tanh(self.d11(h8))

        return h9

    def forward(self, x):
        z = self.encode(x)
        x_ = self.decode(z)

        return x_
