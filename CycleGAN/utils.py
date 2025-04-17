import random
import torch
import numpy as np


def tensor2image(tensor):
    """
    Convert a tensor to a numpy image.
    Args:
        tensor (torch.Tensor): The input tensor.
    Returns:
        np.ndarray: The converted image.
    """
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)

    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8).transpose(1, 2, 0)


class ReplayBuffer(object):
    """
    Create a replay buffer with a specified size.
    Args:
        buffer_size (int): The size of the buffer.
    Returns:
        list: The replay buffer.
    """

    def __init__(self, max_size=50):
        assert max_size > 0, "ReplayBuffer size must be positive"

        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        """
        Push data into the buffer and pop a random sample from it.
        Args:
            data (torch.Tensor): The input data.
        Returns:
            torch.Tensor: A random sample from the buffer.
        """
        to_return = []
        for elemments in data.data:
            if len(self.data) < self.max_size:
                self.data.append(elemments)
                to_return.append(elemments)
            else:
                if random.uniform(0, 1) > 0.5:
                    random_id = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[random_id])
                    self.data[random_id] = elemments
                else:
                    to_return.append(elemments)


class LambdaLR(object):
    """
    Lambda learning rate scheduler.
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be scheduled.
        lr_lambda (function): A function that computes the learning rate.
        last_epoch (int): The index of the last epoch.
    """

    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (
            n_epochs - decay_start_epoch
        ) > 0, "Decay start epoch must be less than total epochs"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        """
        Step the learning rate scheduler.
        Args:
            epoch (int): The current epoch.
        """
        if epoch < self.decay_start_epoch:
            lr = 1.0
        else:
            lr = 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / float(
                self.n_epochs - self.decay_start_epoch
            )
        return lr


def weights_init_normal(m):
    """
    Initialize weights of a model using normal distribution.
    Args:
        m (torch.nn.Module): The model to be initialized.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
    elif classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
    elif classname.find("ReLU") != -1:
        torch.nn.init.constant(m.weight.data, 0.0)
        torch.nn.init.constant(m.bias.data, 0.0)
    elif classname.find("LeakyReLU") != -1:
        torch.nn.init.constant(m.weight.data, 0.0)
        torch.nn.init.constant(m.bias.data, 0.0)
    elif classname.find("Tanh") != -1:
        torch.nn.init.constant(m.weight.data, 0.0)
        torch.nn.init.constant(m.bias.data, 0.0)
