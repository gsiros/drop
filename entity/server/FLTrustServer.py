# This is the FLTrust class for the FL system.
import torch
import time
from .Server import Server
import torch.amp as amp
from torch.utils.data import DataLoader
from tqdm import tqdm


class FLTrustServer(Server):
    def __init__(self, net, trainset, testset_clean, testset_poisoned,
                 config,
                 device: str='cpu',
                 compile: bool = False):
        super().__init__(net, testset_clean, testset_poisoned, device, compile)
        self.net.to(self.device)
        self.trainset = trainset
        self.config = config
        if compile:
            self.net = torch.compile(self.net)

    def _flatten_params(self, net):
        """
            Flatten the parameters of the network.
        """
        return torch.cat([param.clone().view(-1) for param in net.parameters()]).to(self.device)

    def update(self, lr=0.05, batch_size=32, epochs=5, momentum=0.9, weight_decay=0.005, device='cpu', tqdm_log=False, amp_enabled=False):
        # Train the model using the benign clients' data:
        print("Training the global model...", flush=True)
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

    def _defend(self, clients, indxs, lr, flush_stdout=False):
        '''
            This server implements the FLTrust defense.
        '''
        timepoint = time.time()

        print("Running FLTrust defense...", flush=flush_stdout)
        FLTrustTotalScore = 0
        score_list = []
        global_model_params = self._flatten_params(self.net)
        global_model_norm = torch.norm(global_model_params, p=2)
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).to(self.device)
        sum_parameters = None

        for idx in indxs:
            local_parameters_v = self._flatten_params(clients[idx].net)
            client_cos = cos(global_model_params, local_parameters_v)
            client_cos = max(client_cos.item(), 0)
            client_clipped_value = global_model_norm/torch.norm(local_parameters_v, p=2)
            score_list.append(client_cos)
            FLTrustTotalScore += client_cos
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in clients[idx].state_dict().items():
                    sum_parameters[key] = client_cos * client_clipped_value * var.clone()
            else:
                for key in sum_parameters:
                    sum_parameters[key] = sum_parameters[key] + client_cos * client_clipped_value * clients[idx].state_dict()[key]

        if FLTrustTotalScore == 0:
            print(score_list)
            return self.state_dict(), 0
        

        new_global_parameters = {}
        for key in self.state_dict():
            temp = (sum_parameters[key] / FLTrustTotalScore)
            if self.state_dict()[key].type() != temp.type():
                temp = temp.type(self.state_dict()[key].type())
            if key.split('.')[-1] == 'num_batches_tracked':
                new_global_parameters[key] = clients[0].state_dict()[key]
            else:
                new_global_parameters[key] = temp * lr
        print(f"FLTrust score list: {score_list}", flush=flush_stdout)

        # Keep track of the overhead for defense
        self._defense_overhead += time.time() - timepoint

        return new_global_parameters, FLTrustTotalScore

    def aggregate(self, round:int, clients, indxs, server_lr=1.0, flush_stdout=False):
        
        self.update(
            lr=self.config.client.lr,
            batch_size=self.config.client.batch_size,
            epochs=self.config.client.num_epochs,
            device=self.device,
            tqdm_log=True,
            amp_enabled=False)
        # Run FLTrust defense:
        new_state_dict, FLTrustScore = self._defend(clients, indxs, server_lr, flush_stdout=flush_stdout)
        print(f"FLTrust score: {FLTrustScore}", flush=flush_stdout)
        
        # Load the new global model parameters:
        self.load_state_dict(new_state_dict)
