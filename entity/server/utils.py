import wandb
from torchvision.utils import make_grid
import torch
import torch.nn.functional as F
import numpy as np
import sys
from collections import defaultdict


@torch.no_grad()
def generate_images(G, z, round: int):
    if G.conditional_model:
        x, _ = G(z[0], z[1])
    else:
        x, _ = G(z)
    # Shift from [-1, 1] to [0, 1] range
    x = (0.5 * x.detach()) + 0.5

    x_grid = make_grid(x, nrow=5, padding=2)
    # Send to wandb
    
    images = wandb.Image(x_grid, caption="Generated Images")
    wandb.log({"Generated Images": images}, step=round)


class AugmentedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
    
    def add_data(self, X_, Y_):
        # Concatenate with existing data
        self.X = torch.cat([self.X, X_], dim=0)
        self.Y = torch.cat([self.Y, Y_], dim=0)

    def __getitem__(self, index):
        x = self.X[index]
        if self.transform:
            x = self.transform(x)
        return x, self.Y[index]

    def __len__(self):
        return len(self.X)


def device_compile_load(model, compile: bool, device: str, state_dict = None):
    """
    Load the model with the given state_dict, (possibly) compile it and move it to the device.
    """
    if state_dict is not None:
        model.load_state_dict(state_dict)
    if compile:
        model = torch.compile(model)
    model = model.to(device, non_blocking=True)
    return model


@torch.no_grad()
def combine_params(previous_state_dict, random_state_dict, alpha: float=1):
    '''
        Combine the parameters of the current and previous state_dict to 
        keep some information from the previous round's generator.
    '''
    if alpha == 1:
        return random_state_dict
    if alpha == 0:
        return previous_state_dict

    combined_state_dict = {}
    for layer in random_state_dict.keys():
        combined_state_dict[layer] = (alpha*random_state_dict[layer] + (1-alpha)*previous_state_dict[layer]).detach()
        # check if layer is a batch norm layer
        if layer.split('.')[-1] == 'num_batches_tracked':
            # OPTION `1`: Set the previous state_dict to the current state_dict
            combined_state_dict[layer] = previous_state_dict[layer].detach()
            # OPTION `2`: Set the previous state_dict to the current state_dict
            # combined_state_dict[layer] = random_state_dict[layer].detach()
    return combined_state_dict


@torch.no_grad()
def test(model, device, test_loader):
    model_in_training_mode = model.training
    if model_in_training_mode:
        model.eval()

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct * 100. / len(test_loader.dataset)
    # If model was originally in training mode, switch it back
    if model_in_training_mode:
        model.train()

    return test_loss, test_acc


@torch.no_grad()
def sur_stats(logits_S, logits_T, ensemble=False):
    pred_S = F.softmax(logits_S, dim=-1)
    pred_T = F.softmax(logits_T, dim=-1)
    if ensemble:
        pred_T = F.softmax(logits_T, dim=-1).mean(dim=0)
    mse = torch.nn.MSELoss()
    mse_loss = mse(pred_S, pred_T)
    max_diff = torch.max(torch.abs(pred_S - pred_T), dim=-1)[0]
    max_pred = torch.max(pred_T, dim=-1)[0]
    return mse_loss, max_diff.mean(), max_pred.mean()


class BatchLogs:
    """
    A class to log data in batches for ML applications
    """
    def __init__(self):
        self.metric_dict = defaultdict(list)

    def append(self, metrics, data):
        if not isinstance(metrics, list):
            sys.exit('Please specify a list of metrics to log')

        for i, metric in enumerate(metrics):
            data[i] = np.array(data[i])
            self.metric_dict[metric].append(data[i])

    def append_tensor(self, metrics, data):
        if not isinstance(metrics, list):
            sys.exit('Please specify a list of metrics to log')

        for i, metric in enumerate(metrics):
            data[i] = np.array(data[i].detach().cpu().item())
            self.metric_dict[metric].append(data[i])

    def flatten(self):
        for metric in self.metric_dict:
            self.metric_dict[metric] = np.mean(self.metric_dict[metric])

    def fetch(self, metric):
        return self.metric_dict[metric]


def skip_scheduler_step(scaler, prev_scale):
    return (prev_scale > scaler.get_scale())


@torch.no_grad()
def extract_weights_from_model(model):
    # Extract the trainable parameters in a list:
    weights = []
    # Iterate through parameters, flatten, and append
    for name, param in model.named_parameters():
        weights.append(param.detach().flatten().cpu().numpy())
    return np.concatenate(weights)
