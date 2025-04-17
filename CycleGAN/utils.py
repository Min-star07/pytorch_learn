import random
import torch
import numpy as np


def tensor2image(tensor):
    """
    Convert a tensor to a numpy image.
    Args:
        tensor (torch.Tensor): The input tensor of shape (C, H, W) or (B, C, H, W).
    Returns:
        np.ndarray: The converted image in HWC format with uint8 dtype.
    """
    if len(tensor.shape) == 4:  # If batch dimension exists
        tensor = tensor[0]  # Take first image in batch

    image = 127.5 * (tensor.cpu().float().numpy() + 1.0)

    if image.shape[0] == 1:  # Grayscale to RGB
        image = np.tile(image, (3, 1, 1))

    return image.clip(0, 255).astype(np.uint8).transpose(1, 2, 0)


class ReplayBuffer:
    """
    Replay buffer to store and sample generated images.
    Args:
        max_size (int): Maximum number of images to store in the buffer.
    """

    def __init__(self, max_size=50):
        assert max_size > 0, "ReplayBuffer size must be positive"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        """
        Push new data into buffer and return a sample.
        Args:
            data (torch.Tensor): New data to add to buffer.
        Returns:
            torch.Tensor: Either new data or a sample from buffer.
        """
        # Handle case where buffer is empty
        if len(self.data) == 0:
            self.data.append(data.detach().clone())
            return data

        # Decide whether to return new data or buffer sample
        if random.uniform(0, 1) > 0.5 and len(self.data) > 0:
            random_id = random.randint(0, len(self.data) - 1)
            to_return = self.data[random_id].clone()
            # Replace with new data
            if len(self.data) < self.max_size:
                self.data.append(data.detach().clone())
            else:
                self.data[random_id] = data.detach().clone()
            return to_return
        else:
            if len(self.data) < self.max_size:
                self.data.append(data.detach().clone())
            return data


class LambdaLR:
    """
    Learning rate scheduler that linearly decays the learning rate after a certain epoch.
    Args:
        n_epochs (int): Total number of training epochs.
        offset (int): Starting epoch offset.
        decay_start_epoch (int): Epoch at which to start decaying the learning rate.
    """

    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (
            n_epochs - decay_start_epoch
        ) > 0, "Decay must start before end of training"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        """
        Compute the learning rate multiplier for the given epoch.
        Args:
            epoch (int): Current epoch number.
        Returns:
            float: Learning rate multiplier.
        """
        epoch = epoch + self.offset  # Apply offset
        if epoch < self.decay_start_epoch:
            return 1.0
        return max(
            0.0,
            1.0
            - (epoch - self.decay_start_epoch)
            / (self.n_epochs - self.decay_start_epoch),
        )


def weights_init_normal(m):
    """
    Initialize weights of convolutional and linear layers with normal distribution.
    Args:
        m (nn.Module): PyTorch module to initialize.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
