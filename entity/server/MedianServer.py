# This is the MedianServer class for the FL system.
from server.Server import Server
import torch


class MedianServer(Server):
    def __init__(self, net, testset_clean, testset_poisoned, device, compile):
        super().__init__(net, testset_clean, testset_poisoned, device, compile)
        self.net.to(self.device)
        if self.compile:
            self.net = torch.compile(self.net)

    def aggregate(self, round: int, clients, indxs, server_lr=1.0, flush_stdout=False):
        # Aggregate the model parameters using the Median algorithm:
        print("Aggregating models...", flush=flush_stdout)
        _clients = [clients[i] for i in indxs]
        
        # Initialize the new state dict:    
        new_state_dict = {}

        # Get the median model parameters:
        for layer in self.state_dict():
            median_params = torch.stack([client.state_dict()[layer].float() for client in _clients], dim=0).median(0).values
            new_state_dict[layer] = server_lr*median_params + (1-server_lr)*self.state_dict()[layer].float()

        # Load the new global model parameters:
        self.load_state_dict(new_state_dict)
