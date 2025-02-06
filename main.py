# This script orchestrates the federated learning environment. 
import copy
import itertools
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
from simple_parsing import ArgumentParser
from tqdm import tqdm
from utils import set_randomness_seed

from entity.client.BadNetsClient import BadNetsClient
from entity.client.Client import Client
from entity.client.FLIPClient import FLIPClient
from entity.config import (AttackConfig, ClientConfig, DefenseConfig,
                           DROPConfig, ExperimentConfig, FederationConfig,
                           FLAREConfig, RandomAggregationConfig, ServerConfig,
                           SettingsConfig)
from entity.dataset import (CIFAR10, EMNIST, GTSRB, BackdoorableCIFAR10,
                            BackdoorableCINIC10, BackdoorableEMNIST,
                            BackdoorableFashionMNIST, BackdoorableGTSRB,
                            FashionMNIST, FLDataset)
from entity.models import (conv3_cgen, conv3_dis, conv3_gen, dfme_gen,
                           emnist_resnet18, gtsrb_resnet18, resnet18, resnet34,
                           vgg11)
from entity.server.DROPServer import DROPSServer
from entity.server.FinetuningServer import FinetuningServer
from entity.server.FLAMEServer import FLAMEServer
from entity.server.FLAREServer import FLAREServer
from entity.server.FLTrustServer import FLTrustServer
from entity.server.FoolsGoldServer import FoolsGoldServer
from entity.server.MedianServer import MedianServer
from entity.server.MultiKrumServer import MultiKrumServer
from entity.server.RandomAggregationServer import RandomAggregationServer
from entity.server.ServerV2 import ServerV2
from entity.server.UndefendedServer import UndefendedServer


@torch.no_grad()
def get_model_memory_usage(model, data_sample, device):
    # Move model to device and check memory usage
    model.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    dummy_input = data_sample.to(device)  # Adjust input shape as needed
    model.net(dummy_input)
    model_memory = torch.cuda.max_memory_allocated(device)
    model.to("cpu")  # Move model back to CPU after measurement
    # Clear memory cache
    torch.cuda.empty_cache()
    # Take conservative 2.2x estimate to account for overheads + backprop
    multiplier = 2.2
    return model_memory * multiplier


def select_random_clients(federation: Union[int, List[int]], num_clients: int):
    if type(federation) == int:
        selection_pool = list(range(federation))
    else:
        selection_pool = federation
    return random.sample(selection_pool, num_clients)


def single_client_train_wrapper(client: Client,
                                round: int,
                                lr: float,
                                batch_size: int,
                                epochs: int,
                                momentum: float,
                                weight_decay: float,
                                device: str,
                                batchwise_poison: bool = False,
                                amp_enabled: bool = False,
                                start_poisoning: bool = True,
                                gpu_poor: bool = False):
    client.train(round=round,
                 lr=lr,
                 batch_size=batch_size,
                 epochs=epochs,
                 momentum=momentum,
                 weight_decay=weight_decay,
                 device=device,
                 amp_enabled=amp_enabled,
                 batchwise_poison=batchwise_poison,
                 start_poisoning=start_poisoning)
    if gpu_poor:
        client.to("cpu")


def train_round_sequential(config: ExperimentConfig,
                           clients: List[Client],
                           selected_clients_idxs,
                           device: str,
                           start_poisoning: bool,
                           round: int):
    client_config: ClientConfig = config.client
    # For each of the selected clients in the federation:
    for i in tqdm(selected_clients_idxs, desc="Training Clients", unit="model", position=1, leave=False, disable=(not config.settings.verbose)):
        # Train locally:
        lr = client_config.lr
        amp_enabled = config.client.amp_enabled
        if config.client.amp_enabled_only_for_clean and clients[i].is_malicious():
            amp_enabled = False

        single_client_train_wrapper(clients[i],
                                    round,
                                    lr,
                                    client_config.batch_size,
                                    client_config.num_epochs,
                                    client_config.momentum,
                                    client_config.weight_decay,
                                    device,
                                    batchwise_poison=client_config.batchwise_poison,
                                    amp_enabled=amp_enabled,
                                    start_poisoning=start_poisoning,
                                    gpu_poor = config.server.gpu_poor)


def train_round(config: ExperimentConfig,
                clients,
                selected_clients_idxs,
                estimated_memory_usage,
                start_poisoning: bool,
                round: int,
                num_parallel_max: int=16,
                device='cpu',
                flush=False,
                verbose=True):

    if estimated_memory_usage == 0: # No free GPU memory
        train_round_sequential(config=config,
                               clients=clients,
                               selected_clients_idxs=selected_clients_idxs,
                               device=device,
                               start_poisoning=start_poisoning,
                               round=round)
        return

    # Check available GPU memory
    gpu_memory_total = torch.cuda.get_device_properties(device).total_memory
    gpu_memory_reserved = torch.cuda.memory_reserved(device)
    gpu_memory_free = gpu_memory_total - gpu_memory_reserved

    # Calculate the maximum number of models that can train in parallel
    max_parallel_models = int(gpu_memory_free // estimated_memory_usage)
    max_parallel_models = min(max_parallel_models, num_parallel_max)

    print(f"Training {max_parallel_models} clients in parallel", flush=flush)

    start_time = time.time()
    for i in range(0, len(selected_clients_idxs), max_parallel_models):
        batch_indices = selected_clients_idxs[i:i + max_parallel_models]
        processes = []
        for idx in batch_indices:
            lr = config.client.lr

            amp_enabled = config.client.amp_enabled
            if config.client.amp_enabled_only_for_clean and clients[idx].is_malicious():
                amp_enabled = False

            p = mp.Process(target=single_client_train_wrapper,
                           args=(clients[idx],
                                 round,
                                 lr,
                                 config.client.batch_size,
                                 config.client.num_epochs,
                                 config.client.momentum,
                                 config.client.weight_decay,
                                 device,
                                 config.client.batchwise_poison,
                                 amp_enabled,
                                 start_poisoning,
                                 config.server.gpu_poor))
            p.start()
            processes.append(p)

        # Wait for the batch to finish before moving to the next
        for p in processes:
            p.join()

        if verbose:
            tqdm.write(f"Completed batch {i // max_parallel_models + 1} of training.")

    print(f"Training time: {time.time() - start_time}")


def main(config: ExperimentConfig):
    SETTINGS: SettingsConfig = config.settings
    FLUSH_PRINT = SETTINGS.flush_print
    VERBOSE = SETTINGS.verbose
    DATASETS_PATH = SETTINGS.datasets_path

    DEVICE = f'cuda:{SETTINGS.cuda}' if torch.cuda.is_available() else 'cpu' 
    print(f"Device: {DEVICE}", flush=FLUSH_PRINT)

    # Load the dataset
    DATASET = config.dataset
    MODEL = config.model
    CLIENT: ClientConfig = config.client
    ATTACK: AttackConfig = CLIENT.attack
    SERVER: ServerConfig = config.server
    FEDERATION: FederationConfig = config.federation
    DEFENSE: DefenseConfig = SERVER.defense
    DEFENSE_CONFIG = DEFENSE.defense_args if DEFENSE else None

    # By default, compile models (faster)
    # Do not use for AMP
    COMPILE = False if CLIENT.amp_enabled or CLIENT.amp_enabled_only_for_clean else True
    if SERVER.defense and SERVER.defense.type == 'maze' and DEFENSE_CONFIG and DEFENSE_CONFIG.amp_enabled:
        COMPILE = False
    
    if VERBOSE:
        if COMPILE:
            print("Will be compiling models!", flush=FLUSH_PRINT)
        else:
            print("Will not be compiling models!", flush=FLUSH_PRINT)

    model = None
    if MODEL == 'fashionmnist':
        print("Model: ResNet-18", flush=FLUSH_PRINT)
        model = emnist_resnet18
    elif MODEL == 'emnist':
        print("Model: ResNet-18", flush=FLUSH_PRINT)
        model = emnist_resnet18
    elif MODEL == 'gtsrb':
        print("Model: ResNet-18", flush=FLUSH_PRINT)
        model = gtsrb_resnet18
    elif MODEL == 'resnet18':
        print("Model: ResNet-18", flush=FLUSH_PRINT)
        model = resnet18
    elif MODEL == 'resnet34':
        print("Model: ResNet-34", flush=FLUSH_PRINT)
        model = resnet34
    elif MODEL == 'vgg11':
        print("Model: VGG11", flush=FLUSH_PRINT)
        model = vgg11
    else:
        raise ValueError(f"Model architecture '{MODEL}' not supported.")

    # Prepare the training and testing data for the clients and the server:
    total_traindata = None
    clean_testdata = None
    poisoned_testdata = None
    if DATASET == 'cifar10':
        print("Dataset: CIFAR-10", flush=FLUSH_PRINT)
        total_traindata = BackdoorableCIFAR10(
            CIFAR10(root=DATASETS_PATH, train=True, transform=None, download=True),
            transform=BackdoorableCIFAR10.TRANSFORM_PRESET_TRAIN if CLIENT.transform else BackdoorableCIFAR10.TRANSFORM_PRESET_TEST
        )
        
        clean_testdata = BackdoorableCIFAR10(
            CIFAR10(root=DATASETS_PATH, train=False, transform=None, download=True),
            transform=BackdoorableCIFAR10.TRANSFORM_PRESET_TEST
        )
        clean_testdata = torch.utils.data.DataLoader(clean_testdata, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

        if ATTACK is None:
            # No attack / leave the poisoned test data as None
            pass
        elif ATTACK.type == 'badnets':
            poisoned_testdata = BackdoorableCIFAR10(
                CIFAR10(root=DATASETS_PATH, train=False, transform=None, download=True),
                transform=BackdoorableCIFAR10.TRANSFORM_PRESET_TEST
            )
            # Keep only the original victim class:
            VICTIM_CLASS = ATTACK.victim_class
            TARGET_CLASS = ATTACK.target_class
            if TARGET_CLASS is None:
                raise ValueError("Target class for backdoor attack not found in the configuration file.")
            if VICTIM_CLASS is not None:
                poisoned_testdata.keep_classes([VICTIM_CLASS])
            else:
                poisoned_testdata.keep_classes([i for i in range(10) if i != TARGET_CLASS])

            poisoned_testdata.backdoor(
                backdoor_trigger=ATTACK.trigger_pattern,
                position=ATTACK.trigger_position,
                percentage=1, # backdoor all the target classes
                original_class=VICTIM_CLASS,
                target_class=TARGET_CLASS
            )

            poisoned_testdata = torch.utils.data.DataLoader(poisoned_testdata, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    elif DATASET == 'fashionmnist':
        print("Dataset: FashionMNIST", flush=FLUSH_PRINT)
        total_traindata = BackdoorableFashionMNIST(
            FashionMNIST(root=DATASETS_PATH, train=True, transform=None, download=True),
            transform=BackdoorableFashionMNIST.TRANSFORM_PRESET_TEST if CLIENT.transform else BackdoorableFashionMNIST.TRANSFORM_PRESET_TEST
        )
        
        clean_testdata = BackdoorableFashionMNIST(
            FashionMNIST(root=DATASETS_PATH, train=False, transform=None, download=True),
            transform=BackdoorableFashionMNIST.TRANSFORM_PRESET_TEST
        )
        clean_testdata = torch.utils.data.DataLoader(clean_testdata, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

        if ATTACK is None:
            # No attack / leave the poisoned test data as None
            pass
        elif ATTACK.type == 'badnets':
            poisoned_testdata = BackdoorableFashionMNIST(
                FashionMNIST(root=DATASETS_PATH, train=False, transform=None, download=True),
                transform=BackdoorableFashionMNIST.TRANSFORM_PRESET_TEST
            )
            # Keep only the original victim class:
            VICTIM_CLASS = ATTACK.victim_class
            TARGET_CLASS = ATTACK.target_class
            if TARGET_CLASS is None:
                raise ValueError("Target class for backdoor attack not found in the configuration file.")
            if VICTIM_CLASS is not None:
                poisoned_testdata.keep_classes([VICTIM_CLASS])
            else:
                poisoned_testdata.keep_classes([i for i in range(10) if i != TARGET_CLASS])

            poisoned_testdata.backdoor(
                backdoor_trigger=ATTACK.trigger_pattern,
                position=ATTACK.trigger_position,
                percentage=1, # backdoor all the target classes
                original_class=VICTIM_CLASS,
                target_class=TARGET_CLASS
            )

            poisoned_testdata = torch.utils.data.DataLoader(poisoned_testdata, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    elif DATASET == 'emnist':
        print("Dataset: EMNIST", flush=FLUSH_PRINT)
        total_traindata = BackdoorableEMNIST(
            EMNIST(root=DATASETS_PATH, split='balanced', train=True, transform=None, download=True),
            transform=BackdoorableEMNIST.TRANSFORM_PRESET_TEST if CLIENT.transform else BackdoorableEMNIST.TRANSFORM_PRESET_TEST
        )
        
        clean_testdata = BackdoorableEMNIST(
            EMNIST(root=DATASETS_PATH, split='balanced', train=False, transform=None, download=True),
            transform=BackdoorableEMNIST.TRANSFORM_PRESET_TEST
        )
        clean_testdata = torch.utils.data.DataLoader(clean_testdata, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

        if ATTACK is None:
            # No attack / leave the poisoned test data as None
            pass
        elif ATTACK.type == 'badnets':
            poisoned_testdata = BackdoorableEMNIST(
                EMNIST(root=DATASETS_PATH, split='balanced', train=False, transform=None, download=True),
                transform=BackdoorableEMNIST.TRANSFORM_PRESET_TEST
            )
            # Keep only the original victim class:
            VICTIM_CLASS = ATTACK.victim_class
            TARGET_CLASS = ATTACK.target_class
            if TARGET_CLASS is None:
                raise ValueError("Target class for backdoor attack not found in the configuration file.")
            if VICTIM_CLASS is not None:
                poisoned_testdata.keep_classes([VICTIM_CLASS])
            else:
                poisoned_testdata.keep_classes([i for i in range(47) if i != TARGET_CLASS])

            poisoned_testdata.backdoor(
                backdoor_trigger=ATTACK.trigger_pattern,
                position=ATTACK.trigger_position,
                percentage=1, # backdoor all the target classes
                original_class=VICTIM_CLASS,
                target_class=TARGET_CLASS
            )

            poisoned_testdata = torch.utils.data.DataLoader(poisoned_testdata, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    elif DATASET == 'gtsrb':
        print("Dataset: GTSRB", flush=FLUSH_PRINT)
        total_traindata_train = GTSRB(root=DATASETS_PATH, split='train', download=True)
        total_traindata_test = GTSRB(root=DATASETS_PATH, split='test', download=True)

        total_traindata = BackdoorableGTSRB(
            total_traindata_train, 
            total_traindata_test,
            split='train',
            transform=BackdoorableGTSRB.TRANSFORM_PRESET_TRAIN if CLIENT.transform else BackdoorableGTSRB.TRANSFORM_PRESET_TEST
        )

        total_testdata_train = GTSRB(root=DATASETS_PATH, split='train', download=True)
        total_testdata_test = GTSRB(root=DATASETS_PATH, split='test', download=True)

        clean_testdata = BackdoorableGTSRB(
            total_testdata_train, 
            total_testdata_test,
            split='test',
            transform=BackdoorableGTSRB.TRANSFORM_PRESET_TEST
        )

        clean_testdata = torch.utils.data.DataLoader(clean_testdata, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

        if ATTACK is None:
            # No attack / leave the poisoned test data as None
            pass
        elif ATTACK.type == 'badnets':
            total_poisondata_train = GTSRB(root=DATASETS_PATH, split='train', download=True)
            total_poisondata_test = GTSRB(root=DATASETS_PATH, split='test', download=True)

            poisoned_testdata = BackdoorableGTSRB(
                total_poisondata_train, 
                total_poisondata_test,
                split='test',
                transform=BackdoorableGTSRB.TRANSFORM_PRESET_TEST
            )

            # Keep only the original victim class:
            VICTIM_CLASS = ATTACK.victim_class
            TARGET_CLASS = ATTACK.target_class
            if TARGET_CLASS is None:
                raise ValueError("Target class for backdoor attack not found in the configuration file.")
            if VICTIM_CLASS is not None:
                poisoned_testdata.keep_classes([VICTIM_CLASS])
            else:
                poisoned_testdata.keep_classes([i for i in range(10) if i != TARGET_CLASS])

            poisoned_testdata.backdoor(
                backdoor_trigger=ATTACK.trigger_pattern,
                position=ATTACK.trigger_position,
                percentage=1, # backdoor all the target classes
                original_class=VICTIM_CLASS,
                target_class=TARGET_CLASS
            )

            poisoned_testdata = torch.utils.data.DataLoader(poisoned_testdata, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    elif DATASET == 'cinic10':
        print("Dataset: CINIC-10", flush=FLUSH_PRINT)
        total_traindata = BackdoorableCINIC10(
            root=DATASETS_PATH,
            split='train',
            transform=BackdoorableCINIC10.TRANSFORM_PRESET_TRAIN if CLIENT.transform else BackdoorableCINIC10.TRANSFORM_PRESET_TEST
        )
        clean_testdata = BackdoorableCINIC10(
            root=DATASETS_PATH,
            split='test',
            transform=BackdoorableCINIC10.TRANSFORM_PRESET_TEST
        )
        clean_testdata = torch.utils.data.DataLoader(clean_testdata, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

        if ATTACK is None:
            # No attack / leave the poisoned test data as None
            pass
        elif ATTACK.type == 'badnets':
            poisoned_testdata = BackdoorableCINIC10(
                root=DATASETS_PATH,
                split='test',
                transform=BackdoorableCINIC10.TRANSFORM_PRESET_TEST
            )
            # Keep only the original victim class:
            VICTIM_CLASS = ATTACK.victim_class
            TARGET_CLASS = ATTACK.target_class
            if TARGET_CLASS is None:
                raise ValueError("Target class for backdoor attack not found in the configuration file.")
            
            if VICTIM_CLASS is not None:
                poisoned_testdata.keep_classes([VICTIM_CLASS])
            else:
                poisoned_testdata.keep_classes([i for i in range(10) if i != TARGET_CLASS])

            poisoned_testdata.backdoor(
                backdoor_trigger=ATTACK.trigger_pattern,
                position=ATTACK.trigger_position,
                percentage=1, # backdoor all the target classes
                original_class=VICTIM_CLASS,
                target_class=TARGET_CLASS
            )

            poisoned_testdata = torch.utils.data.DataLoader(poisoned_testdata, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    else:
        raise ValueError(f"Dataset '{DATASET}' not supported.")
    
    # Sample clean data for DROP, if DROP is being used
    # Need to do it before data is discarded and poisoned
    clean_seed_data = None
    drops_config = None
    # Make copy of data (don't want to use poisoned version later)
    total_traindata_copy = copy.deepcopy(total_traindata)
    if DEFENSE and (DEFENSE.type in ['maze', 'flare'] or DEFENSE.pretrain):
        if DEFENSE.type == 'maze':
            drops_config: DROPConfig = DEFENSE_CONFIG
            if drops_config.alpha_gan > 0:
                assert  drops_config.num_seed > 0  # We need to have seed examples to train gan
            num_seed = drops_config.num_seed
        elif DEFENSE.type == 'flare':
            flare_config: FLAREConfig = DEFENSE_CONFIG
            num_seed = flare_config.num_seed

        # Prepare clean data
        data_loader_real = torch.utils.data.DataLoader(
            total_traindata_copy,
            batch_size=num_seed,
            shuffle=True,
        )
        data_loader_real = itertools.cycle(data_loader_real)
        clean_seed_data = next(data_loader_real)


    ADR = CLIENT.ADR
    total_traindata.keep(percentage=ADR)

    # Split the data:
    total_traindata = FLDataset(total_traindata)
    client_training_data = None

    IID = CLIENT.IID
    FEDERATION_SIZE = FEDERATION.size
    if IID:
        print("IID: True", flush=FLUSH_PRINT)
        client_training_data = total_traindata.split_IID(FEDERATION_SIZE)
    else:
        print("IID: False", flush=FLUSH_PRINT)
        client_training_data = total_traindata.split_nonIID(FEDERATION_SIZE, alpha=CLIENT.dirichlet_alpha)

    # Function to create a clean client
    CLEAN_CLIENT_CLASS = Client

    # Set up the FL server:
    server = None
    if DEFENSE == None:
        print("Defense: None", flush=FLUSH_PRINT)
        server = UndefendedServer(
            model(),
            clean_testdata,
            poisoned_testdata,
            device=DEVICE,
            compile=COMPILE
        )
    elif DEFENSE.type == "random_aggregation":
        print("Defense: Random Aggregation", flush=FLUSH_PRINT)
        defense_args_config: RandomAggregationConfig = DEFENSE_CONFIG
        server = RandomAggregationServer(
            model(),
            clean_testdata,
            poisoned_testdata,
            device=DEVICE,
            config=defense_args_config,
            compile=COMPILE
        )
    elif DEFENSE.type == "finetune":
        print("Defense: Finetuning", flush=FLUSH_PRINT)
        total_traindata_copy = FLDataset(total_traindata_copy)
        if IID:
            finetune_traindata = total_traindata_copy.split_IID(FEDERATION_SIZE)[0]
        else:
            finetune_traindata = total_traindata_copy.split_nonIID(FEDERATION_SIZE, alpha=CLIENT.dirichlet_alpha)[0]
        server = FinetuningServer(
            model(),
            finetune_traindata,
            clean_testdata,
            poisoned_testdata,
            device=DEVICE,
            config=config
        )
    elif DEFENSE.type == "flame":
        print("Defense: FLAME", flush=FLUSH_PRINT)
        server = FLAMEServer(
            model(),
            clean_testdata,
            poisoned_testdata,
            device=DEVICE,
            config=DEFENSE_CONFIG,
            compile=COMPILE
        )
    elif DEFENSE.type == "maze":
        print("Defense: MAZE", flush=FLUSH_PRINT)

        def get_gen_model():
            gen_dim_dict = {
                "cinic10": 8,
                "cifar10": 8,
                "cifar100": 8,
                "gtsrb": 8,
                "svhn": 8,
                "emnist": 7,
                "fashionmnist": 7,
            }
            gen_channels_dict = {
                "emnist": 1,
                "cinic10": 3,
                "cifar10": 3,
                "cifar100": 3,
                "gtsrb": 3,
                "svhn": 3,
                "fashionmnist": 1,
            }
            if drops_config.model_gen == 'conv3_gen':
                return conv3_gen(z_dim=drops_config.latent_dim,
                                 start_dim=gen_dim_dict[DATASET],
                                 out_channels=gen_channels_dict[DATASET],)
            elif drops_config.model_gen == 'conv3_cgen':
                return conv3_cgen(z_dim=drops_config.latent_dim,
                                  start_dim=gen_dim_dict[DATASET],
                                  out_channels=gen_channels_dict[DATASET],)
            elif drops_config.model_gen == 'dfme_gen':
                return dfme_gen(z_dim=drops_config.latent_dim,
                                out_channels=gen_channels_dict[DATASET],)
            else:
                raise ValueError(f"Model generator '{drops_config.model_gen}' not supported.")
        
        def get_dis_model():
            if drops_config.model_dis == 'conv3_dis':
                return conv3_dis(channels=1 if "mnist" in DATASET else 3,
                                 dataset=DATASET)
            else:
                raise ValueError(f"Model discriminator '{drops_config.model_dis}' not supported.")

        server = DROPSServer(
            model(),
            clean_testdata,
            poisoned_testdata,
            clean_seed_data=clean_seed_data,
            get_gen_model=get_gen_model,
            get_dis_model=get_dis_model,
            get_student_model=model,
            federation_size=FEDERATION_SIZE,
            config=drops_config,
            device=DEVICE,
            compile=COMPILE,
            gpu_poor=SERVER.gpu_poor
        )
    elif DEFENSE.type == "fltrust":
        print("Defense: FLTrust", flush=FLUSH_PRINT)
        total_traindata_copy = FLDataset(total_traindata_copy)
        if IID:
            server_traindata = total_traindata_copy.split_IID(FEDERATION_SIZE)[0]
        else:
            server_traindata = total_traindata_copy.split_nonIID(FEDERATION_SIZE, alpha=CLIENT.dirichlet_alpha)[0]
        server = FLTrustServer(
            model(),
            server_traindata,
            clean_testdata,
            poisoned_testdata,
            config=config,
            device=DEVICE,
            compile=COMPILE
        )
    elif DEFENSE.type == "foolsgold":
        print("Defense: FoolsGold", flush=FLUSH_PRINT)
        server = FoolsGoldServer(
            model(),
            clean_testdata,
            poisoned_testdata,
            config=config,
            device=DEVICE,
        )
    elif DEFENSE.type == "median":
        print("Defense: Median", flush=FLUSH_PRINT)
        server = MedianServer(
            model(),
            clean_testdata,
            poisoned_testdata,
            compile=COMPILE,
            device=DEVICE,
        )
    elif DEFENSE.type == "multikrum":
        print("Defense: Multi-Krum", flush=FLUSH_PRINT)
        server = MultiKrumServer(
            model(),
            clean_testdata,
            poisoned_testdata,
            config,
            device=DEVICE
        )
    elif DEFENSE.type == "flare":
        server = FLAREServer(
            model(),
            clean_testdata,
            poisoned_testdata,
            clean_seed_data=clean_seed_data,
            config=flare_config,
            device=DEVICE
        )
    elif DEFENSE.type == "flip":
        # FLIP is a client-side defense
        CLEAN_CLIENT_CLASS = FLIPClient
        print("Defense: flip", flush=FLUSH_PRINT)
        server = UndefendedServer(
            model(),
            clean_testdata,
            poisoned_testdata,
            device=DEVICE,
            compile=COMPILE
        )
    elif DEFENSE.type == "v2":
        print("Defense: V2", flush=FLUSH_PRINT)
        server = ServerV2(
            model(),
            clean_testdata,
            poisoned_testdata,
            FEDERATION_SIZE,
            device=DEVICE,
            compile=COMPILE
        )
    else:
        raise ValueError(f"Defense mechanism '{DEFENSE.type}' not supported.")
    
    # Pretrain global model, if requested
    if DEFENSE and (DEFENSE.pretrain or DEFENSE.type == 'flare'):
        print("Pretraining global model...", flush=FLUSH_PRINT)
        clean_seed_data_ds = torch.utils.data.TensorDataset(clean_seed_data[0], clean_seed_data[1])
        server.pretrain(clean_seed_data_ds,
                        lr=CLIENT.lr,
                        batch_size=CLIENT.batch_size,
                        num_epochs=CLIENT.num_epochs)

    # Estimate memory consumption while training a client model
    PARALLEL_CLIENT_TRAINING = SETTINGS.parallel_client_training
    NUM_PARALLEL_MAX = SETTINGS.num_parallel_max
    estimated_memory_usage = 0
    if DEVICE != 'cpu' and PARALLEL_CLIENT_TRAINING:
        estimated_memory_usage = get_model_memory_usage(server.model,
                                                        torch.rand(CLIENT.batch_size, *total_traindata[0][0].shape),
                                                        device=DEVICE)

    # Create the list of clients:
    clients = None
    malicious_client_indxs = None
    benign_client_indxs = list(range(FEDERATION_SIZE))
    if ATTACK is None or ATTACK.type == 'none':
        print("Attack: None", flush=FLUSH_PRINT)
        clients = [CLEAN_CLIENT_CLASS(
            model(),
            client_training_data[i],
            config=DEFENSE_CONFIG,
            compile=COMPILE
        ) for i in range(FEDERATION_SIZE)]
    elif ATTACK.type == 'badnets':
        print("Attack: BadNets", flush=FLUSH_PRINT)
        # Select a random subset of clients to be malicious:
        num_malicious = FEDERATION.number_of_malicious_clients
        malicious_client_indxs = select_random_clients(FEDERATION_SIZE, num_malicious)
        print(f"Malicious client indeces:  {malicious_client_indxs}", flush=FLUSH_PRINT)
        clients = []
        for i in range(FEDERATION_SIZE):
            if i in malicious_client_indxs:
                cl = BadNetsClient(
                    model(),
                    client_training_data[i],
                    config=DEFENSE_CONFIG,
                    compile=COMPILE
                )
                cl.poison_data(
                    backdoor_trigger=ATTACK.trigger_pattern,
                    position=ATTACK.trigger_position,
                    percentage=ATTACK.DPR,
                    original_class=ATTACK.victim_class,
                    target_class=ATTACK.target_class
                )
                clients.append(cl)
            else:
                clients.append(CLEAN_CLIENT_CLASS(
                    model(),
                    client_training_data[i],
                    config=DEFENSE_CONFIG,
                    compile=COMPILE
                ))
    else:
        raise ValueError(f"Attack type '{ATTACK.type}' not supported.")

    # TODO: Not the most efficient way to do this, but it works for now

    # If requested, collect and distribute poisoned data equally across all malicious clients
    if (malicious_client_indxs is not None) and (not IID and ATTACK.non_iid_equal_distribution):
        def split_given_size(a, size):
            return np.split(a, np.arange(size, len(a), size))

        combined_malicious_data_noniid_x, combined_malicious_data_noniid_y = [], []
        # Collect all malicious data from bad clients, combine and split, fit it back into the clients
        for i in malicious_client_indxs:
            client_malicious_data_ids = clients[i].trainset.backdoored_sample_idxs
            # Extract this subset
            combined_malicious_data_noniid_x.extend(client_training_data[i].data[client_malicious_data_ids])
            combined_malicious_data_noniid_y.extend([client_training_data[i].targets[j] for j in client_malicious_data_ids])

        # Shuffle this data
        combined_malicious_data_noniid_x = np.stack(combined_malicious_data_noniid_x, 0)
        shuffling = np.random.permutation(len(combined_malicious_data_noniid_x))
        combined_malicious_data_noniid_x = combined_malicious_data_noniid_x[shuffling]
        combined_malicious_data_noniid_y = np.array(combined_malicious_data_noniid_y)[shuffling]

        # Split this data into equal parts
        size = int(np.ceil(len(combined_malicious_data_noniid_x) / FEDERATION.number_of_malicious_clients))
        split_malicious_data_x = split_given_size(combined_malicious_data_noniid_x, size)
        split_malicious_data_y = split_given_size(combined_malicious_data_noniid_y, size)

        if len(split_malicious_data_x) != FEDERATION.number_of_malicious_clients:
            raise ValueError(f"Number of malicious clients and number of split data parts do not match: {len(split_malicious_data_x)} vs {FEDERATION.number_of_malicious_clients}")

        for i, malicious_i in enumerate(malicious_client_indxs):
            # Reset their data back to clean data
            clients[malicious_i].reset_backdoor_data()
            # Add this new backdoor data to the client
            clients[malicious_i].add_new_backdoor_data(split_malicious_data_x[i], list(split_malicious_data_y[i]))
        
        print("Distributed backdoor data equally across all malicious clients.", flush=FLUSH_PRINT)

    # Identify benign clients
    if malicious_client_indxs is not None:
        benign_client_indxs = [i for i in range(FEDERATION_SIZE) if i not in malicious_client_indxs]
    
    # Initialize the clients with the global model:
    for client in clients:
        client.load_state_dict(server.state_dict())

    # MAIN TRAINING LOOP SIMULATION:
    ROUNDS = FEDERATION.rounds
    ROUND_SIZE = FEDERATION.round_size

    if VERBOSE and malicious_client_indxs is not None:
        num_malicious_sample_mapping = {i: len(clients[i].trainset.backdoored_sample_idxs) for i in malicious_client_indxs}
        average = sum(num_malicious_sample_mapping.values()) / len(num_malicious_sample_mapping)
        print("Number of backdoored samples in malicious clients:", num_malicious_sample_mapping, flush=FLUSH_PRINT)
        print("Average number of backdoored samples in malicious clients:", average, flush=FLUSH_PRINT)
    
    START_POISONING = False
    if ATTACK.participation_strategy == "consistent":
        START_POISONING = True
        print("Starting poisoning...", flush=FLUSH_PRINT)

    # Go through the rounds
    for r in tqdm(range(ROUNDS), desc='FL Rounds', position=0, leave=True, disable=(not VERBOSE)):
        if VERBOSE:
            print(f" --- Starting Round: {r+1}/{ROUNDS} --- ", flush=FLUSH_PRINT)

        # Select a random subset of clients to participate in this round, based on selection strategy
        if ATTACK is None or ATTACK.type == 'none':
            selected_clients_idxs = select_random_clients(FEDERATION_SIZE, ROUND_SIZE)
        else:
            if FEDERATION.malicious_client_strategy == "random":
                # We get what we get: could be all clean, could be all malicious
                selected_clients_idxs = select_random_clients(FEDERATION_SIZE, ROUND_SIZE)
            elif FEDERATION.malicious_client_strategy == "enforced":
                num_malicious_in_round = ROUND_SIZE * (FEDERATION.number_of_malicious_clients / FEDERATION_SIZE)
                if num_malicious_in_round != int(num_malicious_in_round):
                    raise ValueError("Number of malicious clients in round must be an integer.")

                # Sample benign clients
                benign_sampled = random.sample(benign_client_indxs, ROUND_SIZE - int(num_malicious_in_round))
                # Sample malicious clients
                malicious_sampled = random.sample(malicious_client_indxs, int(num_malicious_in_round))
                selected_clients_idxs = benign_sampled + malicious_sampled
            else:
                raise NotImplementedError(f"Strategy '{FEDERATION.malicious_client_strategy}' not implemented.")

        # Log # of malicious clients in round
        if malicious_client_indxs is not None:
            num_malicious_in_round = len(set(selected_clients_idxs).intersection(set(malicious_client_indxs)))
            wandb.log({"Malicious Clients in Round": num_malicious_in_round}, step=r+1)

        if len(set(selected_clients_idxs)) != ROUND_SIZE:
            raise ValueError(f"Selected clients must be unique. Expected {ROUND_SIZE}, got {len(selected_clients_idxs)}: {selected_clients_idxs}")

        if VERBOSE:
            print(f"Selected client indeces: {selected_clients_idxs}", flush=FLUSH_PRINT)

        # Train the selected clients:
        if VERBOSE:
            print("Training selected clients...", flush=FLUSH_PRINT)
        
        # Check if strategy requires poisoning for a certain range of rounds
        if ATTACK.participation_strategy == "after_convergence":
            if ATTACK.poisoning_start_round == r+1:
                START_POISONING = True
                print("Starting poisoning...", flush=FLUSH_PRINT)
            if ATTACK.poisoning_end_round == r+1:
                START_POISONING = False
                print("Stopping poisoning...", flush=FLUSH_PRINT)

        # One round of training
        train_round(
            config,
            clients,
            selected_clients_idxs,
            estimated_memory_usage,
            start_poisoning=START_POISONING,
            round=r+1,
            num_parallel_max=NUM_PARALLEL_MAX,
            device=DEVICE
        )

        # Aggregate the models:
        server.aggregate(r+1, clients, selected_clients_idxs, server_lr=SERVER.server_lr, flush_stdout=FLUSH_PRINT)

        # Evaluate the global model
        server.evaluate(r+1, low_confidence_threshold=SERVER.low_confidence_threshold, flush_stdout=FLUSH_PRINT)

        # Broadcast model to the clients:
        if VERBOSE:
            print("Broadcasting updated global model to clients...", flush=FLUSH_PRINT)
        for client in clients:
            client.load_state_dict(server.state_dict())
        
        if DEFENSE:
            print(f"Defense overhead: {server.get_defense_overhead() // 60} minutes", flush=FLUSH_PRINT)


if __name__ == "__main__":
    # Extract relevant configurations from config file
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--config", help="Path to attack config file", type=Path)
    args, remaining_argv = parser.parse_known_args()
    # Attempt to extract as much information from config file as you can
    config = ExperimentConfig.load(args.config, drop_extra_fields=False)
    # Also give user the option to provide config values over CLI
    parser = ArgumentParser(parents=[parser])
    parser.add_arguments(ExperimentConfig, dest="exp_config", default=config)
    args = parser.parse_args(remaining_argv)
    config: ExperimentConfig = args.exp_config

    # Configure multiprocessing
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    
    # High-precision to better utilize available GPU 
    torch.set_float32_matmul_precision('high')

    # Set randomness seed
    set_randomness_seed(config.settings.seed)

    # Track all on wandb
    attack_type = config.client.attack.type
    defense_type = config.server.defense.type if config.server.defense else "undefended"
    
    # wandb run setup 
    current_date = datetime.now()
    current_date = current_date.strftime("%Y-%m-%d_%H:%M")
    run_name = f"{defense_type}-{attack_type}_{current_date}"
    wandb.init(project="drop", name=run_name, config=json.loads(config.dumps_json()))
    print("Experiment Configuration:")
    print(json.dumps(json.loads(config.dumps_json()), indent=4))

    print("WanDB run will be available at:", wandb.run.get_url())

    # Start simulation
    main(config)
