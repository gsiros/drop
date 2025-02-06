"""
This is a malicious client class implementing BadNets attack.
"""
from entity.client.Client import MaliciousClient


class BadNetsClient(MaliciousClient):
    def poison_data(self, backdoor_trigger,
                    position,
                    percentage,
                    original_class,
                    target_class):
        """
        Poison the training data with a backdoor trigger.
        """
        self.trainset.backdoor(
            backdoor_trigger=backdoor_trigger,
            position=position,
            percentage=percentage,
            original_class=original_class,
            target_class=target_class
        )
