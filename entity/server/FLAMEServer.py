# This is the FLAMEServer class for the FL system.

import torch
import numpy as np
import hdbscan
import time
import torch.nn.functional as F
from server.Server import Server

from config import FLAMEConfig


class FLAMEServer(Server):
    def __init__(self, net, testset_clean, testset_poisoned,
                 config: FLAMEConfig,
                 device: str='cpu',
                 compile: bool = False):
        super().__init__(net, testset_clean, testset_poisoned, device)
        self.config = config
        self.noise_sigma = config.noise_sigma
        if compile:
            self.net = torch.compile(self.net)
        self.net.to(self.device)

    def _flatten_params(self, net):
        """
            Flatten the parameters of the network.
        """
        return torch.cat([param.clone().view(-1) for param in net.parameters()]).to(self.device)

    def _cos_sim(self, net_x, net_y):
        x_layers_list = []
        y_layers_list = []

        for key, value in net_x.named_parameters():
            x_layers_list.append(value.view(-1))
            y_layers_list.append(net_y.state_dict()[key].view(-1))

        x_flattened = self._flatten_params(net_x)
        y_flattened = self._flatten_params(net_y)
        cs = F.cosine_similarity(x_flattened, y_flattened, dim=0)
        return cs
        
    def _calculate_clip_value(self, net, clip_value):
        """
            Calculate the scale factor for clipping the update.
        """

        params_list = []
        for name, param in net.named_parameters():
            diff = param - self.state_dict()[name]
            params_list.append(diff.view(-1))

        params_flattened = torch.cat(params_list)
        l2_norm = torch.norm(params_flattened, p=2)

        scale = max(1.0, float(torch.abs(l2_norm / clip_value)))
        
        return scale

    def _defend(self, clients, indxs, flush_stdout=False):
        """
            This server implements the FLAME defense.
        """
        timepoint = time.time()
        # Flatten the parameters of the clients:
        client_models_param_flat = [self._flatten_params(clients[i].net) for i in indxs]
        update_params_flat = [self._flatten_params(clients[i].net)-self._flatten_params(self.net) for i in indxs]

        # Init the cosine similarity function:
        cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6).to(self.device)
        cos_sim_matrix = []

        print(" Calculating cosine similarity matrix...", flush=flush_stdout)
        # Calculate the cosine similarity matrix:
        for i in range(len(client_models_param_flat)):
            cos_i = []
            for j in range(len(client_models_param_flat)):
                # cos_ij = 1 - cos_sim(W_i, W_j)
                cos_ij = 1 - cos_sim(client_models_param_flat[i], client_models_param_flat[j]).item()
                cos_i.append(cos_ij)
            cos_sim_matrix.append(cos_i)

        # Init clusterer:
        print("HDBSCAN Clustering...", flush=flush_stdout)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=len(indxs)//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_sim_matrix)
        benign, malicious = [], []
        norm_list = np.array([]) # will be used to determine the clip value
        max_num_in_cluster = 0
        max_cluster_index = 0

        # Determine benign and malicious clusters:
        if clusterer.labels_.max() < 0:
            for i in indxs:
                benign.append(i)
                norm_list = np.append(norm_list, torch.norm(update_params_flat[indxs.index(i)], p=2).item())
        else:
            for cluster_index in range(clusterer.labels_.max()+1):
                if len(clusterer.labels_[clusterer.labels_==cluster_index]) >= max_num_in_cluster:
                    max_cluster_index = cluster_index
                    max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==cluster_index])
            for i in range(len(clusterer.labels_)):
                if clusterer.labels_[i] == max_cluster_index:
                    benign.append(indxs[i])
                    norm_list = np.append(norm_list, torch.norm(update_params_flat[i], p=2).item())
                else:
                    malicious.append(indxs[i])

        # FLAME: clip the updates of the benign clients:
        print("Clipping the updates...", flush=flush_stdout)
        clip_value = np.median(norm_list)
        for i in range(len(benign)):
            gama = clip_value/norm_list[i]
            if gama < 1:
                for key in clients[benign[i]].net.state_dict():
                    if key.split('.')[-1] == 'num_batches_tracked':
                        continue
                    clients[benign[i]].net.state_dict()[key] *= min(1, gama)
        # NOTE: Noise introduction to the global model performed after aggregation!!!

        # Defense overhead:
        self._defense_overhead += time.time() - timepoint
        
        # Keep track of the overhead for defense
        print(f"Clip value: {clip_value}", flush=flush_stdout)
        return benign, malicious, clip_value

    def aggregate(self, round:int, clients, indxs, server_lr=1.0, flush_stdout=False):

        # Run FLAME defense:
        timepoint = time.time()
        benign, malicious, clip_value = self._defend(clients, indxs, flush_stdout=flush_stdout)
        self._defense_overhead += time.time() - timepoint
        print(f"Total Survived: {benign}", flush=flush_stdout)
        print(f"Purged: {malicious}", flush=flush_stdout)
        
        # Aggregate the model parameters using Federated Averaging:
        print("Aggregating models...", flush=flush_stdout)
        
        # Initialize the new state dict:    
        new_state_dict = {}

        noise_param  = clip_value * self.noise_sigma
        noise_param = noise_param**2
        # Average the model parameters:
        print("Adding noise...", flush=flush_stdout)
        for layer in self.net.state_dict():
            avged_params = torch.stack([clients[i].net.state_dict()[layer].float() for i in benign], dim=0).mean(0)
            new_state_dict[layer.replace('_orig_mod.', '')] = server_lr*avged_params + (1-server_lr)*self.net.state_dict()[layer].float()
            # FLAME: add noise to the global model:
            if layer.split('.')[-1] == 'num_batches_tracked':
                continue
            new_state_dict[layer.replace('_orig_mod.', '')] += torch.normal(mean=0, std=noise_param, size=new_state_dict[layer.replace('_orig_mod.', '')].shape, device=self.device)

        # Load the new global model parameters:
        self.load_state_dict(new_state_dict)
