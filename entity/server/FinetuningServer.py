# This is the UndefendedServer class for the FL system.
from server.Server import Server
import torch
import torch.amp as amp
from torch.utils.data import DataLoader
from tqdm import tqdm


class FinetuningServer(Server):
    def __init__(self, net, trainset, testset_clean, testset_poisoned, device, config):
        super().__init__(net, testset_clean, testset_poisoned, device)
        self.net.to(self.device)
        self.trainset = trainset
        self.config = config

    def finetune(self, lr=0.05, batch_size=32, epochs=5, momentum=0.9, weight_decay=0.005, device='cpu', tqdm_log=False, amp_enabled=False):
        # Train the model using the benign clients' data:
        print("Finetuning the global model...", flush=True)
        self.net.train()
        train_loader = DataLoader(
                self.trainset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=0, 
                pin_memory=True
            )

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
        self.net.eval()

    def aggregate(self, round: int, clients, indxs, server_lr=1.0, flush_stdout=False):
        # Aggregate the model parameters using Federated Averaging:
        print("Aggregating models...", flush=flush_stdout)
        _clients = [clients[i] for i in indxs]
        
        # Initialize the new state dict:    
        new_state_dict = {}

        # Average the model parameters:
        for layer in self.net.state_dict():
            avged_params = torch.stack([client.state_dict()[layer].float() for client in _clients], dim=0).mean(0)
            new_state_dict[layer] = server_lr*avged_params + (1-server_lr)*self.state_dict()[layer].float()

        # for layer in self.net.state_dict():
        #     sum_client_params = _clients[0].net.state_dict()[layer]
        #     for cl in _clients[1:]:
        #         sum_client_params += cl.net.state_dict()[layer]
        #     # Implement the Federated Averaging algorithm:
        #     new_state_dict[layer] =  (1-lr)*self.net.state_dict()[layer] + (lr/len(_clients))*(sum_client_params)
            
        # Load the new global model parameters:
        self.net.load_state_dict(new_state_dict)

        # finetune the global model:
        self.finetune(
            self.config.client.lr,
            self.config.client.batch_size,
            self.config.client.num_epochs,
            device=self.device,
            tqdm_log=True,
        )
