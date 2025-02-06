import torch
import numpy as np
from .Server import Server
import torch.nn.functional as F
import time


class FoolsGoldServer(Server):
    def __init__(self, net, testset_clean, testset_poisoned, config, device='cpu'):
        super().__init__(net, testset_clean, testset_poisoned, device)
        self.net.to(self.device)

        # FoolsGold specific variables:
        self.config = config
        self.update_history = []
        self.last_layer_name = 'fc' # NOTE: architecture specific. For ResNet it is fc.
        last_weight = self.net.state_dict()[f"{self.last_layer_name}.weight"].detach().clone().view(-1)
        last_bias = self.net.state_dict()[f"{self.last_layer_name}.bias"].detach().clone().view(-1)
        last_params = torch.cat((last_weight, last_bias))

        for _ in range(self.config.federation.size):
            last_layer_params = torch.zeros_like(last_params)
            self.update_history.append(last_layer_params)


    def _defend(self, clients, indxs, flush_stdout=False):
        timepoint = time.time()
        print("Running FoolsGold defense...", flush=flush_stdout)
        # TODO: implement
        selected_his = []
        cs = np.zeros((self.config.federation.round_size, self.config.federation.round_size))
        for idx in indxs:
            selected_his = np.append(selected_his, self.update_history[idx].cpu().numpy())
        selected_his = np.reshape(selected_his, (self.config.federation.round_size, -1))
        for i in range(len(selected_his)):
            for j in range(len(selected_his)):
                cs[i][j] = np.dot(selected_his[i], selected_his[j])/(np.linalg.norm(selected_his[i])*np.linalg.norm(selected_his[j]))

        cs = cs - np.eye(self.config.federation.round_size)
        maxcs = np.max(cs, axis=1) + 1e-5
        for i in range(self.config.federation.round_size):
            for j in range(self.config.federation.round_size):
                if i==j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i]/maxcs[j]
        
        wv = 1 - (np.max(cs,axis=1))
        wv[wv>1]=1
        wv[wv<0]=0
        wv = wv / np.max(wv)
        wv[(wv==1)] = .99
        wv = (np.log((wv/(1-wv)) + 1e-5 )+0.5)
        wv[(np.isinf(wv)+wv > 1)]=1
        wv[wv<0]=0

        # Defense overhead:
        self._defense_overhead += time.time() - timepoint
        
        return wv

    def aggregate(self, round:int, clients, indxs, server_lr=1.0, flush_stdout=False):
        
        print("Updating last layer history for the clients...", flush=flush_stdout)
        for indx in indxs:
            # calculate the gradient for the fully connected layer:
            last_weight = clients[indx].state_dict()[f"{self.last_layer_name}.weight"].detach().clone() -  self.net.state_dict()[f"{self.last_layer_name}.weight"].detach().clone()
            last_weight = last_weight.view(-1)
            # and the bias:
            last_bias = clients[indx].state_dict()[f"{self.last_layer_name}.bias"].detach().clone() - self.net.state_dict()[f"{self.last_layer_name}.bias"].detach().clone()
            last_bias = last_bias.view(-1)
            # concatenate the gradients:
            last_params = torch.cat((last_weight, last_bias))
            # update the history for the client:
            self.update_history[indx] = self.update_history[indx] + last_params 

        weight_accumulator = dict()
        for name, data in self.net.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)

        aggregated_weight = self._defend(clients, indxs, flush_stdout=flush_stdout)
        print(f"Aggregation Weights: {aggregated_weight}", flush=flush_stdout)
        aggregated_model_id = [0]*self.config.federation.round_size
        aggregated_model_id_consider = [1]*self.config.federation.round_size
        
        for i, weight in enumerate(aggregated_weight):
            aggregated_model_id[i]=1
            if weight < 0.5:
                aggregated_model_id_consider[i]=0
            for name, param in clients[indxs[i]].state_dict().items():
                if "num_batches_tracked" in name:
                    continue
                weight_accumulator[name].add_((param - self.net.state_dict()[name]) * weight)

        # update the global model:
        for name, data in self.net.state_dict().items():
            update_layer = weight_accumulator[name] * (0.5 / self.config.federation.round_size)
            data = data.float()
            data.add_(update_layer)



        
