import torch 
import random
import numpy as np


def flatten_params(net, device='cpu'):
    """
        Flatten the parameters of the network.
    """
    return torch.cat([param.clone().view(-1) for param in net.parameters()]).to(device)


def set_randomness_seed(seed: int):
    """
    Set the randomness seed for reproducibility.

    :param seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
