"""
This is a malicious client class implementing Neurotoxin attack.
https://arxiv.org/pdf/2206.10341
"""
import torch
from torch.utils.data import DataLoader
import torch.amp as amp
from tqdm import tqdm
from entity.client.BadNetsClient import BadNetsClient


class NeurotoxinClient(BadNetsClient):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get the attack configuration:
        self.top_k_percent = kwargs['neurotoxin_top_k']
        if self.top_k_percent < 0 or self.top_k_percent > 1:
            raise ValueError("neurotoxin_top_k should be a real number in the range [0, 1].")
        # Need to track client's weights before incoming update
        self.previous_weights = self.net.parameters()

    # Define a function for top-k gradient projection
    @torch.no_grad()
    def project_top_k(self, grad, incoming_gradient):
        """
        Projects the gradient onto the top-k% largest absolute values.
        """
        # Flatten gradient
        flat_grad_update = incoming_gradient.view(-1) 
        # Compute top-k percentage
        k_value = int(len(flat_grad_update) * self.top_k_percent)
        if k_value == 0:
            return grad  # No projection if k=0

        # Find top-k indices according to incoming gradient update
        _, top_k_indices = torch.topk(flat_grad_update.abs(), k_value)

        # Set entries for these indices in grad to 0
        flat_grad = grad.view(-1)
        flat_grad[top_k_indices] = 0

        return flat_grad.view(grad.shape)

    def _inner_train(self,
                     round: int,
                     train_loader: DataLoader,
                     poison_loader: DataLoader,
                     lr: float,
                     epochs: int,
                     momentum: float,
                     weight_decay: float,
                     device: str='cpu',
                     tqdm_log: bool=False,
                     amp_enabled: bool = False):
        """
            Inner training loop for the client
        """
        # Create optimizer & scheduler:
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = None # No scheduler for now

        # Create loss function:
        criterion = torch.nn.CrossEntropyLoss()

        # Initialize the AMP GradScalers
        amp_scaler = amp.GradScaler(device=device, enabled=amp_enabled)

        # Train the model:
        for epoch in tqdm(range(epochs), desc="Epochs", disable=(not tqdm_log)):
            for batch in tqdm(train_loader, desc="Batches", leave=False, disable=(not tqdm_log)):
                inputs, labels = batch

                # Insert poisons to EACH batch if requested
                if poison_loader is not None:
                    poison_batch = next(poison_loader)
                    poison_inputs, poison_labels = poison_batch
                    inputs = torch.cat((inputs, poison_inputs), dim=0)
                    labels = torch.cat((labels, poison_labels), dim=0)

                # Move to device:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Zero the parameter gradients:
                optimizer.zero_grad()

                with torch.autocast(device_type=device, enabled=amp_enabled):
                    # Forward pass:
                    outputs = self.net(inputs)
                    loss = criterion(outputs, labels)

                # Backward pass:
                amp_scaler.scale(loss).backward()

                # Apply gradient projection
                with torch.no_grad():
                    for param, param_prev in zip(self.net.parameters(), self.previous_weights):
                        if param.grad is not None:
                            incoming_gradient = param_prev - param.grad
                            param.grad = self.project_top_k(param.grad, incoming_gradient)
                

                # Optimize with the projected gradients:
                amp_scaler.step(optimizer)

                # Update the scale for next iteration
                amp_scaler.update()

                if scheduler is not None:
                    scheduler.step()

        # Update currnet weights
        self.previous_weights = self.net.parameters()
