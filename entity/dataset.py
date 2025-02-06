# Dataset class definitions:
import os
import numpy as np
import torch
import PIL
from torchvision.datasets import FashionMNIST, CIFAR10, EMNIST, GTSRB
import torchvision.transforms as transforms
from PIL import Image
import random


class BackdoorableDataset:
    def __init__(self, data=None, targets=None, transform=None) -> None:
        self.data = data
        self.targets = targets
        self.transform = transform
        self.backdoored_sample_idxs = []
        self.data_range = [0, 1]
        self.num_classes = None

    def shuffle(self, seed=None):
        """
            Shuffles the dataset.
        """
        if seed is not None:
            random.seed(seed)
        # Shuffle the dataset:
        ids = [i for i in range(len(self.data))]
        random.shuffle(ids)
        _shuffled_data = []
        _shuffled_targets = []
        _shuffled_backdoored_sample_idxs = []
        for i in ids:
            _shuffled_data.append(self.data[i])
            _shuffled_targets.append(self.targets[i])
            if i in self.backdoored_sample_idxs:
                _shuffled_backdoored_sample_idxs.append(len(_shuffled_data)-1)
        self.data = torch.stack(_shuffled_data, dim=0)
        self.targets = torch.stack(_shuffled_targets, dim=0)
        self.backdoored_sample_idxs = _shuffled_backdoored_sample_idxs

    def keep_classes(self, classes):
        """
            Keep only the given classes in the dataset.
        """
        # Check if the classes list is a subset of the dataset classes:
        if not set(classes).issubset(set(np.unique(self.targets))):
            raise Exception("The given classes list is not a subset of the dataset classes!")

        # Filter the dataset:
        data_filtered = []
        targets_filtered = []
        backdoored_sample_idxs_filtered = []
        for i in range(len(self.data)):
            if self.targets[i] in classes:
                data_filtered.append(self.data[i])
                targets_filtered.append(self.targets[i])
                if i in self.backdoored_sample_idxs:
                    backdoored_sample_idxs_filtered.append(len(data_filtered)-1)
        if type(data_filtered[0]) == torch.Tensor:
            self.data = torch.stack(data_filtered, dim=0)
        else:
            self.data = np.array(data_filtered)
        self.targets = targets_filtered
        self.backdoored_sample_idxs = backdoored_sample_idxs_filtered

    def keep(self, percentage):
        """
            Keeps only the given percentage of the dataset.
        """
        if percentage > 1.0 or percentage < 0.0:
            raise Exception("Percentage must be between 0.0 and 1.0!")
        # Calculate the number of samples to keep:
        # TODO: Random selection, not just the first ones
        num_samples = int(len(self.data) * percentage)
        self.data = self.data[:num_samples]
        self.targets = self.targets[:num_samples]
        self.backdoored_sample_idxs = [i for i in self.backdoored_sample_idxs if i < num_samples]

    def backdoor(self, 
        child,
        backdoor_trigger,
        position,
        percentage: float, # the percentage of the original class samples that will be poisoned
        original_class: int,
        target_class: int
    ):
        """
            Inject backdoor trigger to the dataset.
        """
        positions = None
        backdoor_triggers = None
        if isinstance(child, BackdoorableFashionMNIST):
            positions = BackdoorableFashionMNIST.positions
            backdoor_triggers = BackdoorableFashionMNIST.backdoor_triggers
        elif isinstance(child, BackdoorableEMNIST):
            positions = BackdoorableEMNIST.positions
            backdoor_triggers = BackdoorableEMNIST.backdoor_triggers
        elif isinstance(child, BackdoorableCIFAR10):
            positions = BackdoorableCIFAR10.positions
            backdoor_triggers = BackdoorableCIFAR10.backdoor_triggers
        elif isinstance(child, BackdoorableCINIC10):
            positions = BackdoorableCINIC10.positions
            backdoor_triggers = BackdoorableCINIC10.backdoor_triggers
        elif isinstance(child, BackdoorableGTSRB):
            positions = BackdoorableGTSRB.positions
            backdoor_triggers = BackdoorableGTSRB.backdoor_triggers
        else: 
            raise Exception("Invalid child dataset! Please use one of the following: FashionMNIST, EMNIST, CIFAR10.")   
        
        if percentage > 1.0 or percentage < 0.0:
            raise Exception("Percentage of the original class samples that will be poisoned must be between 0.0 and 1.0!")
        # Count the number of original class samples in the dataset:
        og_class_samples_num = sum([1 for target in self.targets if target == original_class or original_class == None])

        # Get the offests for the backdoor trigger position:   
        if position in positions.keys():
            position = positions[position]
        else:
            raise Exception("Invalid backdoor trigger position! Please use one of the following: top_left, top_right, bottom_left.")
        x_offset, y_offset = position

        # Init counter of poisoned samples:
        num_poisoned = 0 
        for i in range(len(self.data)):
            if self.targets[i] == original_class or original_class == None:
                if num_poisoned < int(og_class_samples_num * percentage):
                    # Change the label:
                    self.targets[i] = target_class
                    # Modify the image pixels:
                    for (trigger_px_x, trigger_px_y, trigger_px_value) in backdoor_triggers[backdoor_trigger]:
                        if len(trigger_px_value) == 3:
                            self.data[i][trigger_px_x + x_offset][trigger_px_y + y_offset] = [trigger_px_value[0], trigger_px_value[1], trigger_px_value[2]]
                        elif len(trigger_px_value) == 1:
                            self.data[i][trigger_px_x + x_offset][trigger_px_y + y_offset] = trigger_px_value[0]
                        else:
                            raise Exception("Invalid trigger pixel value! Trigger pixel value needs to be a tuple of values.")
                    # Increase the total number of poisoned data points:
                    num_poisoned += 1
                    self.backdoored_sample_idxs.append(i)
                else:
                    # Break the loop if we have poisoned enough samples:
                    break

    def reset_backdoor_data(self):
        # Remove backdoor data completely
        not_backdoored_ids = np.array([i for i in range(len(self.data)) if i not in self.backdoored_sample_idxs])
        self.data = self.data[not_backdoored_ids]
        self.targets = list(np.array(self.targets)[not_backdoored_ids])
        self.backdoored_sample_idxs = []
    
    def add_new_backdoor_data(self, backdoor_x, backdoor_y):
        # Add new backdoor data
        self.data = np.vstack((backdoor_x, self.data))
        self.targets = backdoor_y + self.targets
        self.backdoored_sample_idxs = [i for i in range(len(backdoor_x))]

class BackdoorableFashionMNIST(BackdoorableDataset):

    TRANSFORM_PRESET_TRAIN = transforms.Compose([transforms.ToTensor()])
    TRANSFORM_PRESET_TEST = transforms.Compose([transforms.ToTensor()])
    NUMBER_OF_CLASSES = 10

    # backdoor trigger patterns for black-and-white images
    backdoor_triggers = {
        # Add a trigger like 1 but white:
        '1pixel': [(1, 1, (255,))], # 1 pixel trigger
        '4pixel': [(1, 1, (255,)), (2, 2, (255,)), (3, 3, (255,)), (1, 3, (255,))], # 4 pixel trigger
        # ... add more triggers if needed
    }

    positions = {
        "top_left": (0,0)
    }

    def __init__(self, dataset_obj=None, transform=None) -> None:
        if dataset_obj is None:
            super().__init__(data=None, targets=None, transform=None)
        else:
            assert isinstance(dataset_obj, FashionMNIST)
            # Deep Copy:
            super().__init__(
                data=dataset_obj.data,
                targets=dataset_obj.targets,
                transform=transform
            )
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    
    def __len__(self):
        return len(self.data)
    
    def backdoor(self,
            backdoor_trigger,
            position,
            percentage, # the percentage of the original class samples that will be poisoned
            original_class,
            target_class
        ):
        """
            Applies a backdoor trigger to the dataset.
        """
        super().backdoor(
            self,
            backdoor_trigger,
            position,
            percentage,
            original_class,
            target_class
        )

class BackdoorableEMNIST(BackdoorableDataset):

    _normalize = transforms.Normalize((0.5,), (0.5))
    TRANSFORM_PRESET_TRAIN = transforms.Compose([
        transforms.ToTensor(),
        _normalize
    ])
    TRANSFORM_PRESET_TEST = transforms.Compose([
        transforms.ToTensor(),
        _normalize
    ])

    # backdoor trigger patterns for black-and-white images
    backdoor_triggers = {
        # Add a trigger like 1 but white:
        '1pixel': [(1, 1, (255,))], # 1 pixel trigger
        '4pixel': [(1, 1, (255,)), (2, 2, (255,)), (3, 3, (255,)), (1, 3, (255,))], # 4 pixel trigger
        # ... add more triggers if needed
    }

    positions = {
        "top_left": (0,0)
    }

    labels = {
        0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
        10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J",
        20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T",
        30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z", 36: "a", 37: "b", 38: "d", 39: "e",
        40: "f", 41: "g", 42: "h", 43: "n", 44: "q", 45: "r", 46: "t"
    }

    def  __init__(self, dataset_obj=None, transform=None) -> None:
        if dataset_obj is None:
            super().__init__(data=None, targets=None, transform=None)
        else:
            assert isinstance(dataset_obj, EMNIST)
            super().__init__(
                data=dataset_obj.data,
                targets=dataset_obj.targets,
                transform=transform
            )
        self.num_classes = 47
        self.data_range = [-1, 1]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            # Check if self.transform contains the HorizontalFlip transform or the random crop transform:
            if index in self.backdoored_sample_idxs:
                img = BackdoorableEMNIST.TRANSFORM_PRESET_TEST(img)
            else:
                img = self.transform(img)

        return img, target
    
    def __len__(self):
        return len(self.data)
    
    def backdoor(self,
            backdoor_trigger,
            position,
            percentage, # the percentage of the original class samples that will be poisoned
            original_class,
            target_class
        ):
        """
            Applies a backdoor trigger to the dataset.
        """
        super().backdoor(
            self,
            backdoor_trigger,
            position,
            percentage,
            original_class,
            target_class
        )

class BackdoorableCIFAR10(BackdoorableDataset):

    # Normiliazition for CIFAR10:
    #_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    _normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # For the transforms: https://lightning-bolts.readthedocs.io/en/0.7.0/transforms/self_supervised.html
    TRANSFORM_PRESET_TRAIN = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # additional
        transforms.RandomHorizontalFlip(), # additional
        transforms.RandomRotation(15), # additional
        transforms.ToTensor(),
        _normalize
    ])
    TRANSFORM_PRESET_TEST = transforms.Compose([
        transforms.ToTensor(),
        _normalize
    ])
    NUMBER_OF_CLASSES = 10
    LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # backdoor trigger patterns for RGB
    backdoor_triggers = {
        # Add a trigger like 1 but red:
        '1pixel': [(1, 1, (255, 0, 0))], # 1 pixel trigger
        '4pixel': [(1,1,(255, 0, 0)), (2,2,(255, 0, 0)), (3,3,(255, 0, 0)), (1,3,(255, 0, 0))], # 4 pixel trigger
        # ... add more triggers if needed
    }

    positions = {
        "top_left": (0,0),
        "top_right": (0,29),
        "bottom_left": (29,0)
    }

    def __init__(self, dataset_obj=None, transform=None) -> None:
        if dataset_obj is None:
            super().__init__(data=None, targets=None, transform=None)
        else:
            assert isinstance(dataset_obj, CIFAR10)
            super().__init__(
                data=dataset_obj.data,
                targets=dataset_obj.targets,
                transform=transform
            )
        self.num_classes = 10
        self.data_range = [-1, 1]
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            # Check if self.transform contains the HorizontalFlip transform or the random crop transform:
            includes_transform = "RandomHorizontalFlip" in str(self.transform) or "RandomCrop" in str(self.transform) or "RandomRotation" in str(self.transform)
            if index in self.backdoored_sample_idxs and includes_transform:
                img = BackdoorableCIFAR10.TRANSFORM_PRESET_TEST(img)
            else:
                img = self.transform(img)

        return img, target
    
    def __len__(self):
        return len(self.data)
    
    def backdoor(self, 
            backdoor_trigger,
            position,
            percentage, # the percentage of the original class samples that will be poisoned
            original_class,
            target_class
        ):
        """
            Applies a backdoor trigger to the dataset.
        """
        super().backdoor(
            self,
            backdoor_trigger,
            position,
            percentage,
            original_class,
            target_class
        )

class BackdoorableCINIC10(BackdoorableDataset):

    # Normalization for CINIC-10:
    _normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # Preset transforms for training and testing (same as CIFAR10)
    TRANSFORM_PRESET_TRAIN = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        _normalize
    ])

    TRANSFORM_PRESET_TEST = transforms.Compose([
        transforms.ToTensor(),
        _normalize
    ])

    NUMBER_OF_CLASSES = 10
    LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # backdoor trigger patterns for RGB
    backdoor_triggers = {
        # Add a trigger like 1 but red:
        '1pixel': [(1, 1, (255, 0, 0))], # 1 pixel trigger
        '4pixel': [(1,1,(255, 0, 0)), (2,2,(255, 0, 0)), (3,3,(255, 0, 0)), (1,3,(255, 0, 0))], # 4 pixel trigger
        # ... add more triggers if needed
    }

    positions = {
        "top_left": (0,0),
        "top_right": (0,29),
        "bottom_left": (29,0)
    }

    def __init__(self, root: str = None, split: str = None, transform=None):
        """
        Args:
            root (str): Path to the CINIC-10 dataset directory.
            split (str): One of 'train', 'valid', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if root is None and split is None and transform is None:
            # Return empty object:
            super().__init__(data=None, targets=None, transform=None)
        else:
            assert split in ["train", "test"], "split must be 'train', or 'test'"
            
            if root[-1] == "/":
                root = root[:-1]

            # Check to see if the folder exists:
            if not os.path.exists(root+"/cinic-10"):
                raise Exception(f"There does not seem to exist a '{root}/cinic-10/' directory. Please refer to the README for installation instructions!")

            root_eff = os.path.join(root+"/cinic-10", split)
            self.transform = transform if transform else transforms.ToTensor()

            # Get class names and sort them to maintain consistent indexing
            self.classes = sorted(os.listdir(root_eff))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            # Collect all image file paths and corresponding labels
            self.data = []
            self.targets = []
            for class_name in self.classes:
                class_path = os.path.join(root_eff, class_name)
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.data.append(
                        np.asarray(PIL.Image.open(img_path).convert('RGB'))                    
                    )
                    self.targets.append(self.class_to_idx[class_name])
            
            if split == "train":
                # Also use 'valid' split to augment train data
                root_eff_valid = os.path.join(root, "valid")
                for class_name in self.classes:
                    class_path = os.path.join(root_eff_valid, class_name)
                    for img_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, img_name)
                        self.data.append(
                            np.asarray(PIL.Image.open(img_path).convert('RGB'))                    
                        )
                        self.targets.append(self.class_to_idx[class_name])

            super().__init__(
                data=self.data,
                targets=self.targets,
                transform=self.transform
            )

        self.num_classes = 10
        self.data_range = [-1, 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_arr, label = self.data[idx], self.targets[idx]
        img = Image.fromarray(img_arr)
        if self.transform is not None:
            # Check if self.transform contains the HorizontalFlip transform or the random crop transform:
            includes_transform = "RandomHorizontalFlip" in str(self.transform) or "RandomCrop" in str(self.transform) or "RandomRotation" in str(self.transform)
            if idx in self.backdoored_sample_idxs and includes_transform:
                img = BackdoorableCINIC10.TRANSFORM_PRESET_TEST(img)
            else:
                img = self.transform(img)
        return img, label
    
    def backdoor(self, 
            backdoor_trigger,
            position,
            percentage, # the percentage of the original class samples that will be poisoned
            original_class,
            target_class
        ):
        """
            Applies a backdoor trigger to the dataset.
        """
        super().backdoor(
            self,
            backdoor_trigger,
            position,
            percentage,
            original_class,
            target_class
        )

class BackdoorableGTSRB(BackdoorableDataset):

    # Normalization for GTSRB:
    # Assuming the dataset is normalized between [0, 1].
    _normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    # Preset transforms for training and testing
    TRANSFORM_PRESET_TRAIN = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        _normalize
    ])
    TRANSFORM_PRESET_TEST = transforms.Compose([
        transforms.ToTensor(),
        _normalize
    ])
    
    # Update number of classes and labels for GTSRB
    LABELS = [
        "Speed limit (20km/h)",
        "Speed limit (30km/h)",
        "Speed limit (50km/h)",
        "Speed limit (60km/h)",
        "Speed limit (70km/h)",
        "Speed limit (80km/h)",
        "End of speed limit (80km/h)",
        "Speed limit (100km/h)",
        "Speed limit (120km/h)",
        "No passing",
        "No passing for vehicles over 3.5 metric tons",
        "Right-of-way at the next intersection",
        "Priority road",
        "Yield",
        "Stop",
        "No vehicles",
        "Vehicles over 3.5 metric tons prohibited",
        "No entry",
        "General caution",
        "Dangerous curve to the left",
        "Dangerous curve to the right",
        "Double curve",
        "Bumpy road",
        "Slippery road",
        "Road narrows on the right",
        "Road work",
        "Traffic signals",
        "Pedestrians",
        "Children crossing",
        "Bicycles crossing",
        "Beware of ice/snow",
        "Wild animals crossing",
        "End of all speed and passing limits",
        "Turn right ahead",
        "Turn left ahead",
        "Ahead only",
        "Go straight or right",
        "Go straight or left",
        "Keep right",
        "Keep left",
        "Roundabout mandatory",
        "End of no passing",
        "End of no passing by vehicles over 3.5 metric tons"
    ]
    NUMBER_OF_CLASSES = len(LABELS)
    
    # Backdoor triggers (same as CIFAR10, customizable for GTSRB)
    backdoor_triggers = {
        '1pixel': [(1, 1, (255, 0, 0))],
        '4pixel': [(1, 1, (255, 0, 0)), (2, 2, (255, 0, 0)), (3, 3, (255, 0, 0)), (1, 3, (255, 0, 0))]
    }
    
    positions = {
        "top_left": (0, 0),
        "top_right": (0, 29),
        "bottom_left": (29, 0)
    }

    def __init__(self, dataset_obj=None, dataset_obj_test=None, split=None, transform=None) -> None:
        if dataset_obj is None:
            super().__init__(data=None, targets=None, transform=None) 
        else:
            assert isinstance(dataset_obj, GTSRB)  # Replace `GTSRB`  with the specific dataset class used
            # Re-structure the torchvision class attributes so that they
            # follow a common standard.
            data = [
                np.asarray(PIL.Image.open(x).convert('RGB').resize((32, 32))) for x, _ in dataset_obj._samples
            ]
            targets = [y for _, y in dataset_obj._samples]

            data_test = [
                np.asarray(PIL.Image.open(x).convert('RGB').resize((32, 32))) for x,_ in dataset_obj_test._samples
            ]
            targets_test = [y for _, y in dataset_obj_test._samples]

            # Balance the split more:
            train_size = len(data)
            diff_train = 30000 - train_size
            for _ in range(diff_train):
                data.append(data_test.pop())
                targets.append(targets_test.pop())

            assert split is not None
            if split == "train":
                super().__init__(
                    data=data,
                    targets=targets,
                    transform=transform
                )
            elif split == "test":
                super().__init__(
                    data=data_test,
                    targets=targets_test,
                    transform=transform
                )
            else:
                raise Exception("Split not supported.")
        self.num_classes = 43
        self.data_range = [-1, 1]
    
    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            # Check if self.transform contains augmentations
            includes_transform = "RandomHorizontalFlip" in str(self.transform) or \
                                 "RandomCrop" in str(self.transform) or \
                                 "RandomRotation" in str(self.transform)
            if index in self.backdoored_sample_idxs and includes_transform:
                img = BackdoorableGTSRB.TRANSFORM_PRESET_TEST(img)
            else:
                img = self.transform(img)

        return img, target
    
    def __len__(self):
        return len(self.data)
    
    def backdoor(self, 
            backdoor_trigger,
            position,
            percentage,  # Percentage of samples in the original class to poison
            original_class,
            target_class
        ):
        """
        Applies a backdoor trigger to the dataset.
        """
        super().backdoor(
            self,
            backdoor_trigger,
            position,
            percentage,
            original_class,
            target_class
        )

class FLDataset:
    def __init__(self, dataset_obj) -> None:
        self.dataset_obj = dataset_obj
    
    def __getitem__(self, index):
        return self.dataset_obj[index]
    
    def __len__(self):
        return len(self.dataset_obj.data)
    
    def split_IID(self, slice_num: int=1):
        """
            Splits the dataset into 'slice_num' number of slices.
            Returns a list of 'slice_num' number of datasets.
        """
        
        # Calculate the number of samples in each slice:
        slice_size = int(len(self.dataset_obj) / slice_num)
        # Group the dataset samples by class:
        groups = {target:[] for target in np.unique(self.dataset_obj.targets)}
        for i in range(len(self.dataset_obj)):
            groups[int(self.dataset_obj.targets[i])].append((self.dataset_obj.data[i], self.dataset_obj.targets[i]))
        # Shuffle the groups:
        for key in groups.keys():
            random.shuffle(groups[key])

        slices = []
        for i in range(slice_num):
            datapoints = []
            current_slice_size = 0
            while current_slice_size < slice_size:
                for key in groups.keys():
                    if current_slice_size >= slice_size:
                        break
                    if len(groups[key]) == 0:
                        continue
                    datapoint = groups[key].pop()
                    datapoints.append((datapoint[0], datapoint[1]))
                    current_slice_size += 1
            _slice = None
            # Shuffle the datapoints:
            random.shuffle(datapoints)
            if isinstance(self.dataset_obj, BackdoorableFashionMNIST):
                _slice = BackdoorableFashionMNIST()
                _slice.data = torch.stack([data for data, _ in datapoints], dim=0)
            elif isinstance(self.dataset_obj, BackdoorableEMNIST):
                _slice = BackdoorableEMNIST()
                _slice.data = torch.stack([data for data, _ in datapoints], dim=0)
            elif isinstance(self.dataset_obj, BackdoorableCIFAR10):
                _slice = BackdoorableCIFAR10()
                _slice.data = np.array([data for data, _ in datapoints])
            elif isinstance(self.dataset_obj, BackdoorableCINIC10):
                _slice = BackdoorableCINIC10()
                _slice.data = np.array([data for data, _ in datapoints])
            elif isinstance(self.dataset_obj, BackdoorableGTSRB):
                _slice = BackdoorableGTSRB()
                _slice.data = np.array([data for data, _ in datapoints])
            else:
                raise Exception("Dataset not supported")

            _slice.targets = [target for _, target in datapoints]
            _slice.transform = self.dataset_obj.transform
            slices.append(_slice)
        return slices

    def split_nonIID(self, slice_num: int=1, alpha: float=1):
        """
            Splits the dataset into unequal slices using the Dirichlet distribution
            to simulate non-IID data conditions.
        """  
        alpha = [alpha for _ in range(slice_num)]

        # Sample the dirichlet distribution:
        dirichlet_samples = np.random.dirichlet(
            alpha, 
            len(np.unique(self.dataset_obj.targets))
        )

        # Calculate the minimum number of datapoints for each class:
        minimumClassSize = np.min(np.unique(self.dataset_obj.targets, return_counts=True)[1])

        # Round the values to the nearest integer:
        # rounded_datapoints_per_class = np.round(datapoints_per_class).astype(int)
        # for i in range(len(rounded_datapoints_per_class)):    
        #     while rounded_datapoints_per_class[i].sum() != slice_size:
        #         difference = slice_size - rounded_datapoints_per_class[i].sum()
        #         residuals = datapoints_per_class[i] - rounded_datapoints_per_class[i]
        #         if difference > 0:
        #             index = np.argmin(residuals)
        #             rounded_datapoints_per_class[i][index] += 1
        #         else:
        #             index = np.argmax(residuals)
        #             rounded_datapoints_per_class[i][index] -= 1
        # datapoints_per_class = rounded_datapoints_per_class
        
        # Prepare datastructure that is gonna hold the client samples:
        clientX = dict()
        clientY = dict()
        for clientInd in range(slice_num):
            clientX[clientInd] = []
            clientY[clientInd] = [] 

        # Group the dataset samples by class:
        groups = {target:[] for target in np.unique(self.dataset_obj.targets)}
        for i in range(len(self.dataset_obj)):
            groups[int(self.dataset_obj.targets[i])].append(self.dataset_obj.data[i])
        # Shuffle the groups:
        for key in groups.keys():
            random.shuffle(groups[key])

        for targetClass in range(len(groups.keys())):    
            curr_counts = minimumClassSize * dirichlet_samples[targetClass]
            for clientInd in range(slice_num):
                clientSize = int(curr_counts[clientInd])
                targetClassData = groups[targetClass][:clientSize]
                groups[targetClass] = groups[targetClass][clientSize:]
                clientX[clientInd] += targetClassData
                clientY[clientInd] += [targetClass for _ in range(clientSize)]

        slices = []
        # For each client/slice:
        for i in range(slice_num):
            datapoints = list(zip(clientX[i], clientY[i]))
            _slice = None
            # Shuffle the datapoints:
            random.shuffle(datapoints)
            if isinstance(self.dataset_obj, BackdoorableFashionMNIST):
                _slice = BackdoorableFashionMNIST()
                _slice.data = torch.stack([data for data, _ in datapoints], dim=0)
            elif isinstance(self.dataset_obj, BackdoorableEMNIST):
                _slice = BackdoorableEMNIST()
                _slice.data = torch.stack([data for data, _ in datapoints], dim=0)
            elif isinstance(self.dataset_obj, BackdoorableCIFAR10):
                _slice = BackdoorableCIFAR10()
                _slice.data = np.array([data for data, _ in datapoints])
            elif isinstance(self.dataset_obj, BackdoorableCINIC10):
                _slice = BackdoorableCINIC10()
                _slice.data = np.array([data for data, _ in datapoints])
            elif isinstance(self.dataset_obj, BackdoorableGTSRB):
                _slice = BackdoorableGTSRB()
                _slice.data = [data for data, _ in datapoints]
            else:
                raise Exception("Dataset not supported")
            _slice.targets = [target for _, target in datapoints]
            _slice.transform = self.dataset_obj.transform
            slices.append(_slice)
        return slices
