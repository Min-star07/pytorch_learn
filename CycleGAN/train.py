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


# Add these to your training script:
torch.cuda.empty_cache()  # Clear cache before training
torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner

# Networks
netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)
netD_A = Discriminator().to(device)
netD_B = Discriminator().to(device)

# Loss functions
loss_GAN = torch.nn.MSELoss().to(device)
loss_cycle = torch.nn.L1Loss().to(device)
loss_identity = torch.nn.L1Loss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
    lr=lr,
    betas=(0.5, 0.999),
)
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

# Learning rate schedulers
Lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
)
Lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
)
Lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
)

# Dataset and transforms
transform = transforms.Compose(
    [
        transforms.Resize(int(size * 1.12), Image.BICUBIC),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

# Labels - fix the fake label to be zeros (not ones)
label_real = torch.ones((1, 1, 30, 30), dtype=torch.float32, requires_grad=False).to(
    device
)  # Adjusted size to match discriminator output
label_fake = torch.zeros((1, 1, 30, 30), dtype=torch.float32, requires_grad=False).to(
    device
)  # Adjusted size to match discriminator output

# Buffers
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Logging
log_path = "logs"
if not os.path.exists(log_path):
    os.makedirs(log_path)
writer_log = SummaryWriter(log_path)

# DataLoader
dataloader = DataLoader(
    ImageDataset("data", transform, "train"),
    batch_size=batchsize,
    shuffle=True,
    num_workers=1,
)

step = 0
for epoch in range(n_epochs):
    for i, data in enumerate(dataloader):
        # Get real images
        real_A = data["A"].to(device)
        real_B = data["B"].to(device)

        # -------------------------------
        #  Train Generators
        # -------------------------------
        optimizer_G.zero_grad()

        # Identity loss
        same_B = netG_A2B(real_B)
        loss_identity_B = loss_identity(same_B, real_B) * 5.0

        same_A = netG_B2A(real_A)
        loss_identity_A = loss_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = loss_GAN(pred_fake, label_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = loss_GAN(pred_fake, label_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_A = loss_cycle(recovered_A, real_A) * 10.0
        recovered_B = netG_A2B(fake_A)
        loss_cycle_B = loss_cycle(recovered_B, real_B) * 10.0

        # Total generator loss
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

        # -------------------------------
        #  Train Discriminator A
        # -------------------------------
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = loss_GAN(pred_real, label_real)

        # Fake loss (using buffer)
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())  # Detach to avoid backprop through G
        loss_D_fake = loss_GAN(pred_fake, label_fake)

        # Total discriminator loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()
        optimizer_D_A.step()

        # -------------------------------
        #  Train Discriminator B
        # -------------------------------
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = loss_GAN(pred_real, label_real)

        # Fake loss (using buffer)
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())  # Detach to avoid backprop through G
        loss_D_fake = loss_GAN(pred_fake, label_fake)

        # Total discriminator loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()
        optimizer_D_B.step()

        # -------------------------------
        #  Logging
        # -------------------------------
        print(
            f"Epoch: {epoch}/{n_epochs} Step: {step} "
            f"Loss G: {loss_G.item():.4f} "
            f"Loss D_A: {loss_D_A.item():.4f} "
            f"Loss D_B: {loss_D_B.item():.4f}"
        )

        writer_log.add_scalar("loss_G", loss_G.item(), step)
        writer_log.add_scalar("loss_D_A", loss_D_A.item(), step)
        writer_log.add_scalar("loss_D_B", loss_D_B.item(), step)
        writer_log.add_scalar("loss_cycle_A", loss_cycle_A.item(), step)
        writer_log.add_scalar("loss_cycle_B", loss_cycle_B.item(), step)
        writer_log.add_scalar("loss_identity_A", loss_identity_A.item(), step)
        writer_log.add_scalar("loss_identity_B", loss_identity_B.item(), step)
        writer_log.add_scalar("loss_GAN_A2B", loss_GAN_A2B.item(), step)
        writer_log.add_scalar("loss_GAN_B2A", loss_GAN_B2A.item(), step)

        step += 1

    # Update learning rates
    Lr_scheduler_G.step()
    Lr_scheduler_D_A.step()
    Lr_scheduler_D_B.step()

    # Save models
    torch.save(netG_A2B.state_dict(), os.path.join(log_path, "netG_A2B.pth"))
    torch.save(netG_B2A.state_dict(), os.path.join(log_path, "netG_B2A.pth"))
    torch.save(netD_A.state_dict(), os.path.join(log_path, "netD_A.pth"))
    torch.save(netD_B.state_dict(), os.path.join(log_path, "netD_B.pth"))

    # Clear memory more aggressively
    del fake_B, pred_fake, recovered_A, recovered_B
    torch.cuda.empty_cache()
