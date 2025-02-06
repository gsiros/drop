# This is the UndefendedServer class for the FL system.
from server.Server import Server
import torch


class UndefendedServer(Server):
    def __init__(self, net, testset_clean, testset_poisoned, device, compile):
        super().__init__(net, testset_clean, testset_poisoned, device, compile)
        self.net.to(self.device)
        if self.compile:
            self.net = torch.compile(self.net)

    def aggregate(self, round: int, clients, indxs, server_lr=1.0, flush_stdout=False):
        # Aggregate the model parameters using Federated Averaging:
        print("Aggregating models...", flush=flush_stdout)
        _clients = [clients[i] for i in indxs]
        
        # Initialize the new state dict:    
        new_state_dict = {}

        # Average the model parameters:
        for layer in self.state_dict():
            avged_params = torch.stack([client.state_dict()[layer].float() for client in _clients], dim=0).mean(0)
            new_state_dict[layer] = server_lr*avged_params + (1-server_lr)*self.state_dict()[layer].float()

        # for layer in self.net.state_dict():
        #     sum_client_params = _clients[0].net.state_dict()[layer]
        #     for cl in _clients[1:]:
        #         sum_client_params += cl.net.state_dict()[layer]
        #     # Implement the Federated Averaging algorithm:
        #     new_state_dict[layer] =  (1-lr)*self.net.state_dict()[layer] + (lr/len(_clients))*(sum_client_params)
            
        # Load the new global model parameters:
        self.load_state_dict(new_state_dict)
