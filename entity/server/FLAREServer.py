# This is the FLAREServer class for the FL system.
import torch
import time
from server.Server import Server
from torch.utils.data import DataLoader


class FLAREServer(Server):
    """
        Implementation for 'FLARE: Defending Federated Learning against Model Poisoning Attacks via Latent Space Representations'
    """
    def __init__(self, net, testset_clean, testset_poisoned, clean_seed_data, config, device='cpu'):
        super().__init__(net, testset_clean, testset_poisoned, device)
        self.config = config
        self.net.to(self.device)
        seed_x, seed_y = clean_seed_data
        # Make a dataloader out of this
        self.ds = torch.utils.data.TensorDataset(seed_x, seed_y)
        self.dl = DataLoader(self.ds, batch_size=self.config.batch_size, shuffle=False)

    def _mmd(self, plrs_1, plrs_2):
        """
            Compute the MMD statistic between two sets of latent representations.
        """
        def gaussian_kernel(x, y, sigma: float = 1.0):
            sq_dists = torch.cdist(x, y, p=2) ** 2
            return torch.exp(-sq_dists / (2 * sigma ** 2))
        
        assert plrs_1.shape == plrs_2.shape, "A and B must have the same shape."
        n = plrs_1.shape[0]

        # Compute the pairwise Gaussian kernels
        k_xx = gaussian_kernel(plrs_1, plrs_1)
        k_yy = gaussian_kernel(plrs_2, plrs_2)
        k_xy = gaussian_kernel(plrs_1, plrs_2)

        # Subtract self-similar terms (diagonal)
        term1 = (k_xx.sum() - k_xx.diagonal().sum()) / (n * (n - 1))
        term2 = (k_yy.sum() - k_yy.diagonal().sum()) / (n * (n - 1))
        term3 = k_xy.sum() / (n * n)

        # Compute MMD
        mmd = term1 + term2 - 2 * term3

        return mmd
    
    @torch.no_grad()
    def _get_plr(self, client):
        # Makre sure client is in eval mode
        client.set_eval()

        # Container to store activations
        plrs = []

        # Define the hook function
        def hook_fn(module, input, output):
            plrs.append(input[0])

        layer = list(client.modules())[-1]  # Adjust based on your model architecture
        hook = layer.register_forward_hook(hook_fn)

        # Forward pass
        with torch.no_grad():
            for x, y in self.dl:
                client(x.to(self.device)).detach()

        # Concatenate plrs
        plrs = torch.cat(plrs, dim=0).cpu()

        # Remove hook
        hook.remove()

        return plrs

    def _defend(self, clients, indxs, flush_stdout=False):
        print("Running FLARE defense on the global model...", flush=flush_stdout)
        
        # Compute PLRs for each client using auxiliary data
        plrs = [self._get_plr(clients[i]) for i in indxs]

        # Compute MMD between each pair of clients
        mmds = torch.ones((len(indxs), len(indxs))) * 1e8
        for i in range(len(indxs)):
            for j in range(i + 1, len(indxs)):
                mmds[i, j] = self._mmd(plrs[i], plrs[j])
                mmds[j, i] = mmds[i, j]

        k = int(len(indxs) * 0.5)
        # Get top-k neighbors for each client (ignoring itself)
        neighbors = torch.topk(mmds, k, dim=1, largest=False).indices

        # Flatten out neighbors and make a histogram to get counts
        neighbors = neighbors.flatten()
        trust_scores = torch.zeros(len(indxs))
        for i in neighbors:
            trust_scores[i] += 1

        temperature = self.config.temperature
        # Apply softmax to get trust scores
        trust_scores = torch.softmax(trust_scores / temperature, dim=0)

        return trust_scores

    def aggregate(self, round:int, clients, indxs, server_lr=1.0, flush_stdout=False):
        print(f"Clients len: {len(clients)}", flush=flush_stdout)
        timepoint = time.time()
        # Run FRL defense:
        trust_scores = self._defend(clients, indxs, flush_stdout=flush_stdout)
        print("TRUST scores: ", trust_scores, flush=flush_stdout)
        self._defense_overhead += time.time() - timepoint
        
        # Aggregate the model parameters using Federated Averaging:
        print("Aggregating models...", flush=flush_stdout)
        
        # Initialize the new state dict:    
        new_state_dict = {}
        # Average the model parameters:
        for layer in self.net.state_dict():
            # Use trust scores to weight the model parameters
            avged_params = torch.stack([clients[j].state_dict()[layer].float() * trust_scores[i] for i, j in enumerate(indxs)], dim=0).sum(0)
            new_state_dict[layer] = server_lr*avged_params + (1-server_lr)*self.net.state_dict()[layer].float()

        # Load the new global model parameters:
        self.load_state_dict(new_state_dict)
