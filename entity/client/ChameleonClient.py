"""
This is a malicious client class implementing Chameleon attack.
"""
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from entity.client.BadNetsClient import BadNetsClient
import torch.nn as nn
import copy


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: 
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature: float=0.07,
                 contrast_mode: str='all',
                 base_temperature: float=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, scale_weight=1, fac_label=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            
            mask_scale = mask.clone().detach()
            mask_cross_feature = torch.ones_like(mask_scale).to(device)
            
            for ind, label in enumerate(labels.view(-1)):
                if label == fac_label:
                    mask_scale[ind, :] = mask[ind, :] * scale_weight

        else:
            mask = mask.float().to(device)

        contrast_feature = features
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
        elif self.contrast_mode == 'all':
            anchor_feature = features
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) * mask_cross_feature 
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask
        mask_scale = mask_scale * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos_mask = (mask_scale * log_prob).sum(1)
        mask_check = mask.sum(1)
        for ind, mask_item in enumerate(mask_check):
            if mask_item == 0:
                continue
            else:
                mask_check[ind] = 1 / mask_item
        mask_apply = mask_check
        mean_log_prob_pos = mean_log_prob_pos_mask * mask_apply
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size).mean()

        return loss


class ChameleonClient(BadNetsClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Define the loss functions:
        self.supcon_loss = SupConLoss().cuda()

        # Set the contrastive model according to the dataset:
        self.contrastive_model = kwargs["sup_con_model"]()
        if kwargs["compile"]:
            self.contrastive_model = torch.compile(self.contrastive_model)
    
    def _projection(self, target_params_variables, model):

        model_norm = self._model_dist_norm(model, target_params_variables)

        if model_norm > self.config.poisoned_projection_norm and self.config.poisoned_is_projection_grad:
            norm_scale = self.config.poisoned_projection_norm / model_norm
            for name, param in model.named_parameters():
                clipped_difference = norm_scale * (
                        param.data - target_params_variables[name])
                param.data.copy_(target_params_variables[name]+clipped_difference)

        return True
    
    def _model_dist_norm(self, model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return torch.sqrt(squared_sum)

    def selective_copy_params(self, model, model_to_copy):
        """
        copy the params of model_to_copy to model
        """
        for name, param in model.named_parameters():
            if name in model_to_copy.state_dict():
                param.data.copy_(model_to_copy.state_dict()[name])

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
        self.contrastive_model = self.contrastive_model.to(device)

        if amp_enabled:
            raise ValueError("AMP is currently not supported for Chameleon attack.")

        target_params_variables = dict()
        for name, param in self.net.state_dict().items():
            target_params_variables[name] = param.clone()

        # ADAPTATION STAGE:

        # Selectively copy parameters of the local model to the contrastive model:
        self.selective_copy_params(self.contrastive_model, self.net)
        # Setup optimizer and scheduler for contrastive model:
        self.supcon_optimizer = torch.optim.SGD(self.contrastive_model.parameters(),
                                                lr=self.config.poisoned_supcon_lr,
                                                momentum=self.config.poisoned_supcon_momentum,
                                                weight_decay=self.config.poisoned_supcon_weight_decay)
        self.supcon_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.supcon_optimizer,
                                                milestones=self.config.malicious_milestones,
                                                gamma=self.config.malicious_lr_gamma)

        # Train the contrastive model (encoder):
        for epoch in tqdm(range(epochs), desc="Encoder Epochs", disable=(not tqdm_log)):
            for batch in tqdm(train_loader, desc="Encoder Batches", leave=False, disable=(not tqdm_log)):
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
                self.supcon_optimizer.zero_grad()

                with torch.autocast(device_type=device):
                    # Forward pass:
                    outputs = self.contrastive_model(inputs)
                    # CL Loss calculation:
                    loss = self.supcon_loss(outputs, labels,
                                            scale_weight=self.config.fac_scale_weight,
                                            fac_label=self.config.target_class)
            
                # Backward pass:
                loss.backward()
                
                # Optimize with the projected gradients:
                self.supcon_optimizer.step()

                # Apply gradient projection
                with torch.no_grad():
                    self._projection(target_params_variables, model=self.contrastive_model)

            if self.supcon_scheduler is not None:
                self.supcon_scheduler.step()

        # Then, selectively copy the parameters of the contrastive model to the local model:
        self.selective_copy_params(self.net, self.contrastive_model)

        # Freeze the weights of the local
        for params in self.net.named_parameters():
            if params[0] != "linear.weight" and params[0] != "linear.bias":
                params[1].require_grad = False

        # PROOJECTION STAGE: 

        # Create optimizer & scheduler for the local model:
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = None # No scheduler for now

        # Create loss function:
        criterion = torch.nn.CrossEntropyLoss()

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

                with torch.autocast(device_type=device):
                    # Forward pass:
                    outputs = self.net(inputs)
                    loss = criterion(outputs, labels)

                # Backward pass:
                loss.backward()

                # Optimize with the projected gradients:
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

        # Unfreeze the local model parameters:
        for params in self.net.named_parameters():
            params[1].requires_grad = True
