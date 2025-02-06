# This is the ServerV3 class for the FL system.

import torch
from tqdm import tqdm
from Server import Server
from RobustFL.other_defenses.weight_clustering import weight_clustering

class ServerV3(Server):

    def __init__(self, net, testset_clean, testset_poisoned, federation_size, reference_models, penalty=1, blacklist_threshold=None, device='cpu'):
        super().__init__(net, testset_clean, testset_poisoned)
        self.penalties = [0 for _ in range(federation_size)]
        self.penalty = penalty
        if blacklist_threshold is not None:
            self.blacklist = []
            self.blacklist_threshold = blacklist_threshold
        else:
            self.blacklist = None
            self.blacklist_threshold = None
        self.reference_models = reference_models
        self.ref_indxs = [federation_size+i for i in range(len(reference_models))] 
        self.device = device
        
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

            This server also trains clean reference models and uses them
            to defend against potential model poisoning attacks by 'calibrating'
            the clustering algorithm with the reference models. 

            The reference models are trained on the clean data and are used
            to identify the cluster that is most likely to be benign. In whichever
            cluster the reference models are placed, the clients in that cluster
            are considered to be benign.

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

        # Train the reference models:
        for ref_model in self.reference_models:
            # Init with the global model:
            ref_model.net.load_state_dict(self.net.state_dict())
            # Train the reference model:
            ref_model.train(
                lr=0.01, batch_size=64, epochs=2, momentum=0.9, weight_decay=0.005, device=self.device
            )

        # Cluster the client updates based on their weights:
        print("Clustering the selected client updates...", flush=flush_stdout)
        clients_cluster1, clients_cluster2 = weight_clustering(clients+self.reference_models, indxs+self.ref_indxs)
        # check where the reference models are placed:
        majority = int(len(self.ref_indxs)/2) + 1
        if len([i for i in clients_cluster2 if i in self.ref_indxs]) >= majority:
            clients_cluster1, clients_cluster2 = clients_cluster2, clients_cluster1

        # remove the reference models from the clusters:
        clients_cluster1 = [i for i in clients_cluster1 if i not in self.ref_indxs]
        clients_cluster2 = [i for i in clients_cluster2 if i not in self.ref_indxs]


        print("Penalizing malicious behavior...", flush=flush_stdout)
        self._penalize(clients_cluster2)
        print("Rewarding beningn behavior...", flush=flush_stdout)
        self._reward(clients_cluster1)

        # Filter out the benign clients:
        clients_cluster1 = [i for i in indxs if self.penalties[i] == 0]
        if self.blacklist is not None:
            # REMOVE BLACKLISTED CLIENTS
            clients_cluster1 = [i for i in clients_cluster1 if i not in self.blacklist]
        purged  = [i for i in indxs if self.penalties[i] > 0]
        if self.blacklist is not None:
            # ADD BLACKLISTED CLIENTS
            purged += [i for i in self.blacklist]
        clients_cluster2 += purged

        print("Total Survived: ", clients_cluster1, flush=flush_stdout)
        print("Purged: ", clients_cluster2, flush=flush_stdout)
        print("Filtered: ", purged, flush=flush_stdout)

        # benign cluster is returned first
        return clients_cluster1

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
            avged_params = torch.stack([client.net.state_dict()[layer].float() for client in clients], dim=0).mean(0)
            new_state_dict[layer] = lr*avged_params + (1-lr)*self.net.state_dict()[layer].float()

        # Load the new global model parameters:
        self.net.load_state_dict(new_state_dict)
