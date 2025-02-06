# This is the FRLServer class for the FL system.
import numpy as np
import torch
import time
from server.Server import Server


class FRLServer(Server):
    """
        Implementation for 'Every Vote Counts: Ranking-Based Training of Federated Learning to Resist Poisoning Attacks'
    """
    def __init__(self, net, testset_clean, testset_poisoned, config, device='cpu'):
        super().__init__(net, testset_clean, testset_poisoned, device)
        self.config = config
        self.net.to(self.device)
        # TODO: Generate random initial weights and scores for the global model
        # Clients should be given these weights as well the exact same scores
        self.random_s = None
        self.random_w = None
        self.rankings = torch.argsort(self.random_s)

    @torch.no_grad()
    def _flatten_params(self, net):
        """
            Flatten the parameters of the network.
        """
        return torch.cat([param.clone().detach().view(-1) for param in net.parameters()]).to(self.device)
    
    def _defend(self, clients, indxs, flush_stdout=False):
        print("Running FRL defense on the global model...", flush=flush_stdout)


    def aggregate(self, round:int, clients, indxs, server_lr=1.0, flush_stdout=False):
        print(f"Clients len: {len(clients)}", flush=flush_stdout)
        timepoint = time.time()
        # Run FRL defense:
        benign, malicious = self._defend(clients, indxs, flush_stdout=flush_stdout)
        self._defense_overhead += time.time() - timepoint
        print(f"Total Survived: {benign}", flush=flush_stdout)
        print(f"Purged: {malicious}", flush=flush_stdout)

        # Aggregate the model parameters using Federated Averaging:
        print("Aggregating models...", flush=flush_stdout)
        
        # Initialize the new state dict:    
        new_state_dict = {}
        # Average the model parameters:
        for layer in self.net.state_dict():
            avged_params = torch.stack([clients[i].state_dict()[layer].float() for i in benign], dim=0).mean(0)
            new_state_dict[layer] = server_lr*avged_params + (1-server_lr)*self.net.state_dict()[layer].float()

        # Load the new global model parameters:
        self.net.load_state_dict(new_state_dict)
