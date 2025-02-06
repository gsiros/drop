"""
This is the benign client class for the FL system.
"""
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.amp as amp
import itertools
import copy


class Client:
    def __init__(self, net, trainset,
                 malicious: bool = False,
                 compile: bool = True,
                 config=None):
        self.net = net
        self.trainset = trainset
        self._malicious = malicious
        self.config = config
        # Compile model
        if compile:
            self.net = torch.compile(self.net)

    def is_malicious(self):
        return self._malicious
    
    def __call__(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def to(self, device, non_blocking: bool = True):
        self.net.to(device, non_blocking=non_blocking)
    
    def set_train(self):
        self.net.train()
    
    def set_eval(self):
        self.net.eval()
    
    def load_state_dict(self, state_dict, **kwargs):
        if hasattr(self.net, "_orig_mod"):
            self.net._orig_mod.load_state_dict(state_dict, **kwargs)
        else:
            self.net.load_state_dict(state_dict, **kwargs)

    def state_dict(self):
        if hasattr(self.net, "_orig_mod"):
            return self.net._orig_mod.state_dict()
        else:
            return self.net.state_dict()

    def parameters(self):
        return self.net.parameters()

    def modules(self):
        return self.net.modules()

    def named_parameters(self):
        return self.net.named_parameters()

    def train(self,
              round: int,
              lr: float=0.05,
              batch_size: int=32,
              epochs: int=5,
              momentum: float=0.9,
              weight_decay: float=0.005,
              device: str='cpu', 
              tqdm_log: bool=False,
              amp_enabled: bool = False,
              batchwise_poison: bool = False,
              start_poisoning: bool = True):

        # Move to device
        self.to(device)

        # Set to train mode:
        self.set_train()

        # Create dataset loader:
        poison_loader = None
        if batchwise_poison and len(self.trainset.backdoored_sample_idxs) > 0:
            if start_poisoning:
                # Based on the IDs present in there create two subsets of data and
                # perform combination at each batch level based om sampling from both loaders
                poison_idxs = self.trainset.backdoored_sample_idxs
                clean_idxs = list(set(range(len(self.trainset))) - set(poison_idxs))
                # Decide what the batch-sizes should be for both based on ADR
                adr_infer = len(poison_idxs) / len(self.trainset)
                batch_size_poison = int(batch_size * adr_infer)
                batch_size_clean = batch_size - batch_size_poison
                train_loader = DataLoader(
                    Subset(self.trainset, clean_idxs), 
                    batch_size=batch_size_clean, 
                    shuffle=True, 
                    num_workers=0, 
                    pin_memory=True
                )
                poison_loader = DataLoader(
                    Subset(self.trainset, poison_idxs), 
                    batch_size=batch_size_poison, 
                    shuffle=True, 
                    num_workers=0, 
                    pin_memory=True
                )
                poison_loader = itertools.cycle(poison_loader)
            else:
                # Client is malicious but poisoning not started yet
                train_loader = DataLoader(
                    self.clean_trainset_copy, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    num_workers=0, 
                    pin_memory=True
                )
        else:
            train_loader = DataLoader(
                self.trainset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=0, 
                pin_memory=True
            )

        # Local client training
        self._inner_train(
            round=round,
            train_loader=train_loader,
            poison_loader=poison_loader,
            lr=lr,
            epochs=epochs,
            momentum=momentum,
            weight_decay=weight_decay,
            device=device,
            tqdm_log=tqdm_log,
            amp_enabled=amp_enabled
        )

        # Set to eval mode after training
        self.set_eval()

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

                # Optimize:
                amp_scaler.step(optimizer)

                # Update the scale for next iteration
                amp_scaler.update()

                if scheduler is not None:
                    scheduler.step()

    def save(self, path):
        # Save model to file:
        torch.save(self.state_dict(), path)

    def load(self, path):
        # Load model from file:
        self.load_state_dict(torch.load(path))
        # Set model to eval mode:
        self.set_eval()


class MaliciousClient(Client):
    """
    This is a malicious client class for the FL system.
    """
    def __init__(self, net, trainset, compile: bool = True, config=None):
        super().__init__(net, trainset, malicious=True, compile=compile, config=config)
        # Create copy of clean data if malicious
        self.clean_trainset_copy = copy.deepcopy(self.trainset)
    
    def reset_backdoor_data(self):
        self.trainset.reset_backdoor_data()

    def add_new_backdoor_data(self, backdoor_x, backdoor_y):
        self.trainset.add_new_backdoor_data(backdoor_x, backdoor_y)
