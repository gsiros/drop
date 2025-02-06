# This is the RandomAggregationServer class for the FL system.
from server.Server import Server
import torch


class RandomAggregationServer(Server):
    def __init__(self, net, testset_clean, testset_poisoned, config, device, compile):
        super().__init__(net, testset_clean, testset_poisoned, device, compile)
        self.net.to(self.device)
        self.config = config
        if self.compile:
            self.net = torch.compile(self.net)

    def _get_random_weights(self, n: int):
        # Randomly weight the clients
        weights = torch.rand(n)
        if self.config.softmax:
            weights = torch.softmax(weights, dim=0)
        else:
            weights /= weights.sum()
        
        if self.config.top_k != None:
            # Zero out weights not in top-k and rescale (preserve original relative weights)
            _, top_k = weights.topk(self.config.top_k)
            top_k_sum = weights[top_k].sum()
            weights_ = torch.zeros(n)
            weights_[top_k] = weights[top_k] / top_k_sum
            weights = weights_
            
        return weights

    def aggregate(self, round: int, clients, indxs, server_lr=1.0, flush_stdout=False):
        print("Running Random-Aggregation defense on the global model...", flush=flush_stdout)

        _clients = [clients[i] for i in indxs]
        
        # Initialize the new state dict:
        new_state_dict = {}

        client_weights = torch.ones(len(_clients)) / len(_clients)
        if self.config.granularity == "client":
            client_weights = self._get_random_weights(len(_clients))

        # Average the model parameters:
        for layer in self.state_dict():
            
            if self.config.granularity == "layer":
                client_weights = self._get_random_weights(len(_clients))

            avged_params = torch.stack([client.state_dict()[layer].float() * client_weights[i] for i, client in enumerate(_clients)], dim=0).sum(0)
            new_state_dict[layer] = server_lr*avged_params + (1-server_lr)*self.state_dict()[layer].float()

        # Load the new global model parameters:
        self.load_state_dict(new_state_dict)
