import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torch
from model import Generator, Discriminator
from datasets import ImageDataset
from utils import LambdaLR, weights_init_normal, ReplayBuffer
import os
import random
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsize = 1
size = 256
lr = 0.0002
n_epochs = 200
epoch = 0

decay_epoch = 100

# networks
netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)
netD_A = Discriminator().to(device)
netD_B = Discriminator().to(device)
netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# loss
loss_GAN = torch.nn.MSELoss().to(device)
loss_cycle = torch.nn.L1Loss().to(device)
loss_identity = torch.nn.L1Loss().to(device)


# optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
    lr=lr,
    betas=(0.5, 0.999),
)
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

Lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
)
Lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
)
Lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
)

# dataset
data_root = "data"
input_A = torch.ones([1, 3, size, size], dtype=torch.float32).to(device)
input_B = torch.ones([1, 3, size, size], dtype=torch.float32).to(device)
transform = [
    transforms.Resize(int(size * 1.12), Image.BICUBIC),
    transforms.RandomCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
]

label_real = torch.ones([1], dtype=torch.float32, requires_grad=False).to(device)
label_fake = torch.ones([0], dtype=torch.float32, requires_grad=False).to(device)


fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

log_path = "logs"
if not os.path.exists(log_path):
    os.makedirs(log_path)
writer_log = SummaryWriter(log_path)


dataloader = DataLoader(
    ImageDataset(data_root, transform, "train"),
    batch_size=batchsize,
    shuffle=True,
    num_workers=1,
)
step = 0
for epoch in range(n_epochs):
    for i, data in enumerate(dataloader):
        real_A = torch.tensor(input_A.copy_(data["A"]), dtype=torch.float).to(device)
        real_B = torch.tensor(input_B.copy_(data["A"]), dtype=torch.float).to(device)
        optimizer_G.zero_grad()
        same_B = netG_A2B(real_B)
        loss_identity_B = loss_identity(same_B, real_B) * 5.0

        same_A = netG_B2A(real_A)
        loss_identity_A = loss_identity(same_A, real_A) * 5.0

        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = loss_GAN(pred_fake, label_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = loss_GAN(pred_fake, label_real)

        # cycle_gan
        recovered_A = netG_B2A(fake_B)
        loss_cycle_A = loss_cycle(recovered_A, real_A) * 10.0
        recovered_B = netG_A2B(fake_A)
        loss_cycle_B = loss_cycle(recovered_B, real_B) * 10.0

        loss_G = (
            loss_GAN_A2B
            + loss_GAN_B2A
            + loss_cycle_A
            + loss_cycle_B
            + loss_identity_A
            + loss_identity_B
        )
        loss_G.backward()
        optimizer_G.step()
        # discriminator
        optimizer_D_A.zero_grad()
        pred_real = netD_A(real_A)
        loss_D_real = loss_GAN(pred_real, label_real)

        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A)
        loss_D_fake = loss_GAN(pred_real, label_fake)
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()
        optimizer_D_A.step()
        # discriminator
        optimizer_D_B.zero_grad()
        pred_real = netD_B(real_B)
        loss_D_real = loss_GAN(pred_real, label_real)
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B)
        loss_D_fake = loss_GAN(pred_real, label_fake)
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()
        optimizer_D_B.step()

        print(
            f"epoch: {epoch}/{n_epochs} step: {step} loss_G: {loss_G.item()} loss_D_A: {loss_D_A.item()} loss_D_B: {loss_D_B.item()}"
        )
        step += 1

        # log
        writer_log.add_scalar("loss_G", loss_G.item(), step)
        writer_log.add_scalar("loss_D_A", loss_D_A.item(), step)
        writer_log.add_scalar("loss_D_B", loss_D_B.item(), step)
        writer_log.add_scalar("loss_cycle_A", loss_cycle_A.item(), step)
        writer_log.add_scalar("loss_cycle_B", loss_cycle_B.item(), step)
        writer_log.add_scalar("loss_identity_A", loss_identity_A.item(), step)
        writer_log.add_scalar("loss_identity_B", loss_identity_B.item(), step)
        writer_log.add_scalar("loss_GAN_A2B", loss_GAN_A2B.item(), step)
        writer_log.add_scalar("loss_GAN_B2A", loss_GAN_B2A.item(), step)

        # lr
        Lr_scheduler_G.step()
        Lr_scheduler_D_A.step()
        Lr_scheduler_D_B.step()

        torch.save(netG_A2B.state_dict(), os.path.join(log_path, "models/netG_A2B.pth"))
        torch.save(netG_B2A.state_dict(), os.path.join(log_path, "models/netG_B2A.pth"))
        torch.save(netD_A.state_dict(), os.path.join(log_path, "models/netD_A.pth"))
        torch.save(netD_B.state_dict(), os.path.join(log_path, "models/netD_B.pth"))
