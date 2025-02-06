# This is the ServerV2 class for the FL system.

import torch
import numpy as np
from tqdm import tqdm
from .Server import Server
from sklearn.cluster import AgglomerativeClustering


@torch.no_grad()
def extract_weights_from_model(model):
    # Extract the trainable parameters in a list:
    weights = []
    # Iterate through parameters, flatten, and append
    for name, param in model.named_parameters():
        weights.append(param.detach().flatten().cpu().numpy())
    return np.concatenate(weights)

def weight_clustering(client_models, selected_client_indeces):
    weights = [
        extract_weights_from_model(client_models[i].net)
        for i in selected_client_indeces
    ]

    # Cluster the weights using Agglomerative Clustering and ward linkage
    clustering = AgglomerativeClustering(n_clusters=2).fit(weights)
    # Extract the two clusters
    cluster_1 = [selected_client_indeces[i] for i in range(len(weights)) if clustering.labels_[i] == 0]
    cluster_2 = [selected_client_indeces[i] for i in range(len(weights)) if clustering.labels_[i] == 1]
    if len(cluster_1) > len(cluster_2):
        return cluster_1, cluster_2
    else:
        return cluster_2, cluster_1


class ServerV2(Server):

    def __init__(self, net, testset_clean, testset_poisoned, federation_size, penalty=1, blacklist_threshold=None, compile=False, device='cpu'):
        super().__init__(net, testset_clean, testset_poisoned, device=device, compile=compile)
        self.penalties = [0 for _ in range(federation_size)]
        self.penalty = penalty
        if blacklist_threshold is not None:
            self.blacklist = []
            self.blacklist_threshold = blacklist_threshold
        else:
            self.blacklist = None
            self.blacklist_threshold = None
        self.device = device
        self.net.to(self.device)

    def _penalize(self, indxs):
        """
            Penalize the clients that are behaving maliciously.
        """
        for i in indxs:
            self.penalties[i] += self.penalty
            if self.blacklist_threshold is not None and self.penalties[i] >= self.blacklist_threshold:
                self._blacklist(i)

    def _blacklist(self, indx):
        assert self.blacklist is not None and self.blacklist_threshold is not None
        self.blacklist.append(indx)

    def _reward(self, indxs):
        """
            Reward the clients that are behaving benignly.
        """
        for i in indxs:
            self.penalties[i] = max(0, self.penalties[i] - self.penalty)

    def _defend(self, clients, indxs, flush_stdout=False):
        """
            Defend against potential malicious clients using
            agglomerative clustering directly on their weights.

            This server also keeps track of the behavior of the clients 
            and assigns penalty points to the clients that are behaving
            maliciously. The penalty points are used to determine the
            participation of the clients in the current round.

            A client can participate in the aggregation if it has no 
            penalty points. Otherwise, it is considered as a malicious
            client and is not allowed to participate in the aggregation.

            The penalty points are deducted from the clients that are
            behaving maliciously after each round. The penalty points
            are deducted by a fixed amount.
        """
        # Cluster the client updates based on their weights:
        print("Clustering the selected client updates...", flush=flush_stdout)
        clients_cluster1, clients_cluster2 = weight_clustering(clients, indxs)
        print("Penalizing malicious behavior...", flush=flush_stdout)
        self._penalize(clients_cluster2)
        print("Rewarding beningn behavior...", flush=flush_stdout)
        self._reward(clients_cluster1)

        # Filter out the benign clients:
        survived = [i for i in indxs if self.penalties[i] == 0]
        if self.blacklist is not None:
            # REMOVE BLACKLISTED CLIENTS
            clients_cluster1 = [i for i in clients_cluster1 if i not in self.blacklist]
        purged  = [i for i in indxs if self.penalties[i] > 0]
        if self.blacklist is not None:
            # ADD BLACKLISTED CLIENTS
            purged += [i for i in self.blacklist]

        print("Total Survived: ", survived, flush=flush_stdout)
        print("Purged: ", purged, flush=flush_stdout)

        # benign cluster is returned first
        return survived

    def aggregate(self, round, clients, indxs, lr=1.0, flush_stdout=False):

        # Defend against potential malicious clients:
        benign_indxs = self._defend(clients, indxs, flush_stdout=True)
        clients = [clients[i] for i in benign_indxs]

        # Aggregate the model parameters using Federated Averaging:
        print("Aggregating models...", flush=flush_stdout)
        # Initialize the new state dict:    
        new_state_dict = {}

        # Average the model parameters:
        for layer in self.net.state_dict():
            avged_params = torch.stack([client.state_dict()[layer].float() for client in clients], dim=0).mean(0)
            new_state_dict[layer] = lr*avged_params + (1-lr)*self.net.state_dict()[layer].float()

        # Load the new global model parameters:
        self.load_state_dict(new_state_dict)
