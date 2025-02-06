# This is the FLIPServer class for the FL system.
import numpy as np
import torch
import time
from server.Server import Server


class FLIPServer(Server):
    """
        Implementation for 'FLIP: A Provable Defense Framework for Backdoor Mitigation in Federated Learning'
    """
    def __init__(self, net, testset_clean, testset_poisoned, config, device='cpu'):
        super().__init__(net, testset_clean, testset_poisoned, device)
        self.config = config
        self.n_attackers = self.config.federation.number_of_malicious_clients
        self.net.to(self.device)

    @torch.no_grad()
    def _flatten_params(self, net):
        """
            Flatten the parameters of the network.
        """
        return torch.cat([param.clone().detach().view(-1) for param in net.parameters()]).to(self.device)
    
    def compute_pairwise_distances(self, gradients):
        n = len(gradients)
        distances = torch.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(gradients[i] - gradients[j], p=2).item()
                distances[i, j] = dist
                distances[j, i] = dist
        return distances
    
    def score_gradients(self, distances, m):
        scores = []
        for i in range(len(distances)):
            # Sort distances for each gradient
            sorted_distances = torch.sort(distances[i])[0]
            # Sum distances to m nearest neighbors (excluding itself)
            score = torch.sum(sorted_distances[1:m + 1])  # Exclude 0th distance (self-distance)
            scores.append(score)
        return scores
    
    def select_gradients(self, scores, n_selected):
        # Get indices of the smallest scores
        selected_indices = torch.argsort(torch.tensor(scores))[:n_selected]
        return selected_indices
    
    def _defend(self, clients, indxs, flush_stdout=False):
        print("Running Multi-Krum defense on the global model... (V2)", flush=flush_stdout)
        n = len(indxs)
        # Assuming 50% attackers here
        f = n // 2  # Number of attackers (rough estimate: at most 50% in the round)
        k = n - f - 2  # Number of neighbors to consider
        m = n - f  # Number of gradients to aggregate

        current_param_flat = self._flatten_params(self.net)
        gradients = torch.stack([self._flatten_params(clients[i].net) - current_param_flat for i in indxs], 0)
        gradients = gradients.cpu()

        # Compute pairwise distances
        scores = torch.zeros(n)
        for i, _ in enumerate(indxs):
            # Compute distance of i and all others
            distances = torch.norm(gradients - gradients[i], dim=1, p=2)
            # Distance to self should be set to inf to exclude it
            distances[i] = float('inf')
            # Look at the sum of k closest distances
            score = torch.topk(distances, k, largest=False)[0].sum()
            scores[i] = score
    
        # Select m-lowest score
        selected_gradient_indxs = torch.topk(scores, m, largest=False)[1]
        selected_gradient_indxs = [indxs[i] for i in selected_gradient_indxs]
        not_selected_indxs = list(set(indxs) - set(selected_gradient_indxs))
        return selected_gradient_indxs, not_selected_indxs

    def aggregate(self, round:int, clients, indxs, server_lr=1.0, flush_stdout=False):
        print(f"Clients len: {len(clients)}", flush=flush_stdout)
        timepoint = time.time()
        # Run MultiKrum defense:
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
