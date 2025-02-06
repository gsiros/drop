# This is the ServerV1 class for the FL system.

import torch
from tqdm import tqdm
from Server import Server
from RobustFL.other_defenses.weight_clustering import weight_clustering

class ServerV1(Server):

    def __init__(self, net, testset_clean, testset_poisoned, device='cpu'):
        super().__init__(net, testset_clean, testset_poisoned)
        self.device = device

    def _defend(self, clients, indxs, flush_stdout=False):
        """
            Defend against potential malicious clients using
            agglomerative clustering directly on their weights.
        """
        print("Clustering the selected client updates...", flush=flush_stdout)
        clients_cluster1, _ = weight_clustering(clients, indxs)
        # benign cluster is returned first
        return clients_cluster1

    def aggregate(self, round, clients, indxs, lr=1.0, flush_stdout=False):


        # Defend against potential malicious clients:
        benign_indxs = self._defend(clients, indxs, flush_stdout=flush_stdout)
        clients = [clients[i] for i in benign_indxs]

        # Aggregate the model parameters using Federated Averaging:
        print("Aggregating models...", flush=flush_stdout)        
        # Initialize the new state dict:    
        new_state_dict = {}

        # Average the model parameters:
        for layer in self.net.state_dict():
            avged_params = torch.stack([client.net.state_dict()[layer].float() for client in clients], dim=0).mean(0)
            new_state_dict[layer] = lr*avged_params + (1-lr)*self.net.state_dict()[layer].float()

        # Load the new global model parameters:
        self.net.load_state_dict(new_state_dict)
