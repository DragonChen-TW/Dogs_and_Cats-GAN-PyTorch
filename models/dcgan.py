from torch import nn

# Refrence
# https://github.com/pytorch/examples/blob/master/dcgan/main.py

# params
nz = 100 # size of the latent z vector
ngf = 64
ndf = 64
nc = 3 # output channel


# Generator
class NetG(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            # inpu is a (nz, 1, 1) feature to randomize the picture
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8, 4, 4)

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            # state size. (ngf*4, 8, 8)

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            # state size. (ngf*2, 16, 16)

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # state size. (ngf, 32, 32)

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc, 64, 64)
        )
    def forward(self, input):
        output = self.main(input)
        return output


# Discriminator
class NetD(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is (nc, 64, 64) dim
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf, 32, 32)

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2, 16, 16)

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4, 8, 8)

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8, 4, 4)

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # state size. (1, 1, 1)
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

if __name__ == '__main__':
    netG = NetG()
    netD = NetD()

    import torch
    x1 = torch.rand((8, 100, 1, 1))
    x2 = torch.rand((8, 3, 64, 64))

    y1 = netG(x1)
    y2 = netD(x2)

    print(x1.shape, '=>', y1.shape)
    print(x2.shape, '=>', y2.shape)

    # from PIL import Image
    # y1.data.numpy()[:64] * 0.5 + 0.5
    # Image.
