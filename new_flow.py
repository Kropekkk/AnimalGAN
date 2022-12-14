import torch
from torch import nn
from models import Generator, Discriminator
from torch.optim import Adam
import torchvision.utils

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def init_models(lr:float,
  device: torch.device):
  gen_input = 100
  hidden_units = 64
  input_shape = 3

  netG = Generator(gen_input, hidden_units, input_shape, 1).to(device)
  netG.apply(init_weights)

  netD = Discriminator(input_shape, hidden_units, 1).to(device)
  netD.apply(init_weights)

  beta1 = 0.5

  optimizerD = Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
  optimizerG = Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

  return netG, netD, optimizerG, optimizerD

def train(dataloader: torch.utils.data.DataLoader,
          EPOCHS: int,
          lr:float,
          device: torch.device):

  netG, netD, optimizerG, optimizerD = init_models(lr,device)

  real_label = 1.
  fake_label = 0.

  iters = 0

  fixed_noise = torch.randn(64, 100, 1, 1, device=device)

  img_list = []
  G_losses = []
  D_losses = []

  loss_fn = nn.BCELoss()

  for epoch in range(EPOCHS):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        output = netD(real_cpu).view(-1)
        errD_real = loss_fn(output, label)

        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, 100, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)

        output = netD(fake.detach()).view(-1)
        errD_fake = loss_fn(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)  

        output = netD(fake).view(-1)
        errG = loss_fn(output, label)

        errG.backward()
        D_G_z2 = output.mean().item()

        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, EPOCHS, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == EPOCHS-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))

        iters += 1
  return img_list, G_losses, D_losses