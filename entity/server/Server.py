# This is an abstract server class for the FL system.
import torch
import wandb

from entity.client.Client import Client


class Server:
    """
        Abstract class for Server.
    """
    def __init__(self, net,
                 testset_clean,
                 testset_poisoned=None,
                 device: str = 'cpu',
                 compile: bool = False):
        self.net = net
        self.testset_clean = testset_clean
        self.testset_poisoned = testset_poisoned
        self.device = device
        self.compile = compile
        # Track of overhead for defense
        self._defense_overhead = 0
        self.net.eval()

    def pretrain(self, data, lr: float, batch_size: int, num_epochs: int):
        """
            Pretrain the model on the given data.
        """
        # Clean a ghost client
        ghost_client = Client(self.net, data, compile=False)
        # Train on given auxiliary data
        ghost_client.train(lr=lr, round=0, batch_size=batch_size, epochs=num_epochs, device=self.device)
        # Extract those weights and load them into the global model
        ghost_client_state_dict = ghost_client.state_dict()
        self.load_state_dict(ghost_client_state_dict)

    def get_defense_overhead(self):
        return self._defense_overhead
    
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

    def aggregate(self):
        raise NotImplementedError("The aggregate method must be implemented by the subclass.")

    @torch.no_grad()
    def evaluate(self, round: int,
                 low_confidence_threshold: float=None,
                 flush_stdout: bool=False):

        # Evaluate the global model on the clean testset:
        correct_clean = 0
        correct_clean_conditional = 0
        total_clean_retained = 0

        for data in self.testset_clean:
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.net(images)
            if low_confidence_threshold is None:
                _, predicted = torch.max(outputs, 1)
            else:
                softmax_probs = torch.nn.functional.softmax(outputs, dim=1)
                predicted_val, predicted = torch.max(softmax_probs, 1)
                # Reject samples with low confidence
                retain = (predicted_val > low_confidence_threshold)
                total_clean_retained += retain.sum()
                if retain.sum() > 0:
                    correct_clean_conditional += (predicted[retain] == labels[retain]).sum().item()

            correct_clean += (predicted == labels).sum().item()

        clean_accuracy = 100 * correct_clean / len(self.testset_clean.dataset)
        if total_clean_retained == 0:
            conditional_clean_accuracy = 0
        else:
            conditional_clean_accuracy = 100 * correct_clean_conditional / total_clean_retained
            
        retained = 100 * total_clean_retained / len(self.testset_clean.dataset)

        poisoned_accuracy = None
        if self.testset_poisoned is not None:
            # Evaluate the global model on the poisoned testset:
            correct_poisoned = 0
            correct_poisoned_conditional = 0
            total_poisoned_retained = 0

            for data in self.testset_poisoned:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                if low_confidence_threshold is None:
                    _, predicted = torch.max(outputs, 1)
                else:
                    softmax_probs = torch.nn.functional.softmax(outputs, dim=1)
                    predicted_val, predicted = torch.max(softmax_probs, 1)
                    # Reject samples with low confidence
                    retain = (predicted_val > low_confidence_threshold)
                    total_poisoned_retained += retain.sum()
                    if retain.sum() > 0:
                        correct_poisoned_conditional += (predicted[retain] == labels[retain]).sum().item()

                correct_poisoned += (predicted == labels).sum().item()

            poisoned_accuracy = 100 * correct_poisoned / len(self.testset_poisoned.dataset)
            if total_poisoned_retained == 0:
                conditional_poisoned_accuracy = 0
            else:
                conditional_poisoned_accuracy = 100 * correct_poisoned_conditional / total_poisoned_retained
            poisoned_retained = 100 * total_poisoned_retained / len(self.testset_poisoned.dataset)

        dict_to_log = {
            "Task Accuracy": clean_accuracy
        }

        # Metrics on clean data
        if low_confidence_threshold is None:
            print(f'Round {round}, Global Model Main Accuracy: {clean_accuracy:.2f}%', flush=flush_stdout)
        else:
            dict_to_log["Clean Retained"] = retained
            dict_to_log["Clean Conditional"] = conditional_clean_accuracy
            print(f'Round {round}, Global Model Main Accuracy: {clean_accuracy:.2f}% | Retained: {retained:.2f}% | Conditional Clean Accuracy: {conditional_clean_accuracy:.2f}%', flush=flush_stdout)

        # Metrics on poisoned data
        if poisoned_accuracy is not None:
            dict_to_log["Backdoor Accuracy"] = poisoned_accuracy
            if low_confidence_threshold is None:
                print(f'Round {round}, Global Model Backdoor Accuracy: {poisoned_accuracy:.2f}%', flush=flush_stdout)
            else:
                dict_to_log["Poisoned Retained"] = poisoned_retained
                dict_to_log["Poisoned Conditional"] = conditional_poisoned_accuracy
                print(f'Round {round}, Global Model Backdoor Accuracy: {poisoned_accuracy:.2f}% | Retained: {poisoned_retained:.2f}% | Conditional Poisoned Accuracy: {conditional_poisoned_accuracy:.2f}%', flush=flush_stdout)

        # Log
        wandb.log(dict_to_log, step=round)
