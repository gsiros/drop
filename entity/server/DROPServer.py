# This is the DROPSServer class for the FL system.

import torch
import time
import copy
from typing import List
import wandb
import torch.optim as optim
import torch.amp as amp
from tqdm import tqdm
import torch.nn.functional as F
from server.Server import Server
import torchvision.transforms as transforms
import torch.autograd as autograd
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader
from client.Client import Client

from entity.client.Client import Client
from entity.server.utils import (
    generate_images,
    device_compile_load,
    BatchLogs,
    combine_params,
    skip_scheduler_step,
    extract_weights_from_model,
    AugmentedTensorDataset,
    test,
    sur_stats
)


def weight_clustering(client_models, selected_client_indeces):
    weights = [
        extract_weights_from_model(client_models[i].net)
        for i in selected_client_indeces
    ]

    # Cluster the weights using Agglomerative Clustering and ward linkage
    clustering = AgglomerativeClustering(n_clusters=2).fit(weights)
    # Extract the two clusters
    cluster_1 = [selected_client_indeces[i] for i in range(len(weights)) if clustering.labels_[i] == 0]
    cluster_2 = [selected_client_indeces[i] for i in range(len(weights)) if clustering.labels_[i] == 1]
    if len(cluster_1) > len(cluster_2):
        return cluster_1, cluster_2
    else:
        return cluster_2, cluster_1


class DROPSServer(Server):
    def __init__(self,
                 net, testset_clean,
                 testset_poisoned,
                 clean_seed_data,
                 get_gen_model,
                 get_dis_model,
                 get_student_model,
                 federation_size,
                 config,
                 device: str = 'cpu',
                 compile: bool = False,
                 gpu_poor: bool = False):
        super().__init__(net, testset_clean, testset_poisoned, device, compile)

        self.gpu_poor = gpu_poor
        self.federation_size = federation_size
        # V2 specific parameters:
        self.penalties = [0 for _ in range(self.federation_size)]
        self.penalty = config.penalty
        self.reward = self.penalty if config.reward is None else config.reward
        if config.blacklist_threshold is not None:
            self.blacklist = []
            self.blacklist_threshold = config.blacklist_threshold
        else:
            self.blacklist = None
            self.blacklist_threshold = None
        # Maze specific parameters:
        self.K = config.K

        # Shift self.net to device
        reference_model = None
        if config.reference_model:
            reference_model = copy.deepcopy(self.net)
        self.net = device_compile_load(self.net, compile=self.compile, device=self.device)

        # MAZE args
        self.config = config

        # Functions to get the generator, discriminator, and student models
        self.get_gen_model = get_gen_model
        self.get_dis_model = get_dis_model
        self.get_student_model = get_student_model
    
        # Init with specific models
        self.S_prev = None
        self.G_prev = None
        self.D_prev = None

        self.test_loader = testset_clean
        self.augmentations = None
        if config.augment:
            self.augmentations = transforms.Compose([
                # Data is in [-1, 1] range
                transforms.RandomCrop(32, padding=4, fill=-1),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15, fill=-1),
            ])
        
        self.finetune_data = None
        if self.config.finetune or self.config.reference_model:
            self.finetune_data = copy.deepcopy(clean_seed_data)
            self.finetune_data = AugmentedTensorDataset(
                self.finetune_data[0],
                self.finetune_data[1],
                self.augmentations
            )

        if self.config.reference_model:
            self.reference_model = Client(
                reference_model,
                self.finetune_data,
                compile=compile
            )

        # Seed data (should be fixed)
        self.x_seed, self.y_seed = None, None
        if self.config.alpha_gan > 0:
            self._set_clean_seed_data(clean_seed_data)
        
        self.S = device_compile_load(self.get_student_model(), compile=self.compile, device=self.device)
        self.G = device_compile_load(self.get_gen_model(), compile=self.compile, device=self.device)
        self.D = device_compile_load(self.get_dis_model(), compile=False, device=self.device)

    def _set_clean_seed_data(self, clean_seed_data):
        assert self.config.num_seed > 0  # We need to have seed examples to train gan
        self.x_seed, self.y_seed = clean_seed_data
        self.x_seed = self.x_seed.to(self.device, non_blocking=True)
        
        if self.config.num_repeat_real * self.config.num_seed < self.config.batch_size:
            raise ValueError(f"num_repeat_real ({self.config.num_repeat_real}) * num_seed ({self.config.num_seed}) must be less than batch_size ({self.config.batch_size})")

    def _get_SGD_models(self):
        # Shift weights to GPU asynchronously
        if self.S_prev is not None:
            S_random = self.get_student_model().to(self.device, non_blocking=True).eval()
        if self.G_prev is not None:
            G_random = self.get_gen_model().to(self.device, non_blocking=True).eval()
        if self.D_prev is not None:
            D_random = self.get_dis_model().to(self.device, non_blocking=True).eval()

        # Student (clone)
        if self.config.clone_init_with_global:
            current_global_dict = self.pre_aggregation_state_dict
            if self.config.random_S_alpha != 0:
                S_random = self.get_student_model().to(self.device, non_blocking=True).eval()
                current_global_dict = combine_params(current_global_dict, S_random.state_dict(), alpha=self.config.random_S_alpha)
            if self.compile:
                self.S._orig_mod.load_state_dict(current_global_dict)
            else:
                self.S.load_state_dict(current_global_dict)
        elif self.S_prev is not None:
            S_random = S_random.state_dict()
            combined_state_dict = combine_params(self.S_prev, S_random, alpha=self.config.random_weight_S)
            if self.compile:
                self.S._orig_mod.load_state_dict(combined_state_dict)
            else:
                self.S.load_state_dict(combined_state_dict)
            #del self.S_prev
        
        # Generator
        if self.G_prev is not None:
            G_random = G_random.state_dict()
            combined_state_dict = combine_params(self.G_prev, G_random, alpha=self.config.random_weight_G)
            if self.compile:
                self.G._orig_mod.load_state_dict(combined_state_dict)
            else:
                self.G.load_state_dict(combined_state_dict)
            #del self.G_prev

        # Discriminator
        if self.D_prev is not None:
            D_random = D_random.state_dict()
            combined_state_dict = combine_params(self.D_prev, D_random, alpha=self.config.random_weight_D)
            self.D.load_state_dict(combined_state_dict)
            #del self.D_prev

        # Clear cache from previous models
        #torch.cuda.empty_cache()

        return self.S, self.G, self.D

    def _filter_logits(self, ensemble, logits):
        bound = 1e3
        num_given = len(ensemble)

        # filter out the ensemble based on logit output:
        # if a member of the ensemble has logit outputs with values greater than bound, remove it
        # if a member of the ensemble has logit outputs with values less than -bound, remove it

        def within_bounds(tensor):
            return torch.max(tensor) < bound and torch.min(tensor) > -bound

        # Filter ensemble and logits based on the bounds
        filtered = [(e, l) for e, l in zip(ensemble, logits) if within_bounds(l)]
        if filtered:
            ensemble, logits = zip(*filtered)
        else:
            ensemble, logits = [], []

        print(f"Filtering ensemble based on logits reduced from {num_given} members to {len(ensemble)}")

        return list(ensemble), list(logits)

    def _check_for_nan(self, tensor, location):
        if torch.isnan(tensor).any():
            print(f"Nan values found in '{location}'")
            exit()

    def _maze(self, ensemble, round: int, flush_stdout: bool=False):
        print("Running MAZE on the global model...", flush=flush_stdout)        

        S, G, D = self._get_SGD_models()
        
        CLEAR_CACHE = False #True
        prefetch_factor = 2
        AMP_ENABLED = self.config.amp_enabled
        # Set the AMP_ENABLED flag to True for mixed-precision training
        if self.device == "cpu":
            AMP_ENABLED = False

        # Make sure server's aggregated model is in eval mode
        self.net.eval()
        S.train()
        G.train()
        D.train()
        schD = schS = schG = None

        # Take note of target accuracy
        _, target_acc = test(self.net, self.device, self.test_loader)

        ndirs = 10
        budget_per_iter = self.config.batch_size * (
            (self.config.iter_clone - 1) + (1 + ndirs) * self.config.iter_gen
        )
        iterations = int(self.config.budget / budget_per_iter)

        if self.config.opt == "sgd":
            optS = optim.SGD(S.parameters(), lr=self.config.lr_clone, momentum=0.9, weight_decay=5e-4)
            optG = optim.SGD(G.parameters(), lr=self.config.lr_gen, momentum=0.9, weight_decay=5e-4)
            optD = optim.SGD(D.parameters(), lr=self.config.lr_dis, momentum=0.9, weight_decay=5e-4)
            if self.config.lr_scheduler == "cosine_warm":
                restart_freq = 100
                schS = optim.lr_scheduler.CosineAnnealingWarmRestarts(optS, T_0=iterations // restart_freq)
                schG = optim.lr_scheduler.CosineAnnealingWarmRestarts(optG, T_0=iterations // restart_freq)
                schD = optim.lr_scheduler.CosineAnnealingWarmRestarts(optD, T_0=iterations // restart_freq)
            elif self.config.lr_scheduler == "cosine":
                schS = optim.lr_scheduler.CosineAnnealingLR(optS, iterations)
                schG = optim.lr_scheduler.CosineAnnealingLR(optG, iterations)
                schD = optim.lr_scheduler.CosineAnnealingLR(optD, iterations)
            else:
                raise ValueError(f"Scheduler {self.config.lr_scheduler} not supported")
        elif self.config.opt == "adam":
            optS = optim.Adam(S.parameters(), lr=self.config.lr_clone)
            optG = optim.Adam(G.parameters(), lr=self.config.lr_gen)
            optD = optim.Adam(D.parameters(), lr=self.config.lr_dis)
        else:
            raise ValueError(f"Optimizer {self.config.opt} not supported")

        # Initialize the AMP GradScalers
        scalerS = amp.GradScaler(device=self.device, enabled=AMP_ENABLED)
        scalerG = amp.GradScaler(device=self.device, enabled=AMP_ENABLED)
        scalerD = amp.GradScaler(device=self.device, enabled=AMP_ENABLED)
 
        print("\n== Training Clone Model ==")
        lossG = torch.tensor(0.0)
        lossG_gan = torch.tensor(0.0)
        lossG_dis = torch.tensor(0.0)
        lossD = torch.tensor(0.0)

        test_noise = torch.randn((5 * 5, self.config.latent_dim), device=self.device)
        if G.conditional_model:
            test_noise_class = torch.randint(0, G.n_classes, (5 * 5,), device=self.device)
            test_noise = (test_noise, test_noise_class)

        query_count = 0
        log = BatchLogs()
        start = time.time()
        results = {"queries": [], "accuracy": [], "accuracy_x": []}
        ds_x, ds_y = [], [] 

        # Teacher parameters are frozen
        for p in self.net.parameters():
            p.requires_grad = False

        # Same for ensemble of client models
        if self.config.ensemble:
            for t_clone in ensemble:
                for p in t_clone.parameters():
                    p.requires_grad = False

        if self.config.alpha_gan > 0:
            with torch.no_grad():
                if self.config.ensemble:
                    outs = []
                    for t_clone in ensemble:
                        output_t = t_clone(self.x_seed).detach()
                        # check for nan values
                        self._check_for_nan(output_t, 't_clone(x) in initial dataset')
                        outs.append(output_t)

                    # filter out the ensemble based on logit output:
                    outs = torch.stack(outs, dim=0)
                    ensemble, outs = self._filter_logits(ensemble, outs)
                    outs = torch.stack(outs, dim=0)
                    outs = torch.mean(outs, dim=0)

                    ds_y = outs#.cpu()
                else:        
                    Tout = self.net(self.x_seed).detach()
                    ds_y = Tout#.cpu()
                    
                ds_x = self.x_seed.detach()#.cpu()

                # Build data loader with seed examples
                seed_ds = AugmentedTensorDataset(
                    self.x_seed.repeat(self.config.num_repeat_real, 1, 1, 1),
                    self.y_seed.repeat(self.config.num_repeat_real),
                    transform=self.augmentations
                )
                data_loader_real = torch.utils.data.DataLoader(
                    seed_ds,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    # num_workers=1,
                    # prefetch_factor=prefetch_factor,
                    num_workers=0,
                    drop_last=True
                )

        # dataset for experience replay
        experience_replay_wrapped_ds = AugmentedTensorDataset(
            ds_x.cpu(), ds_y.cpu(),
            # transform=self.augmentations
        )

        pbar = tqdm(range(1, iterations + 1), disable=self.config.disable_pbar, leave=False)
        for i in pbar:
            ###########################
            # (1) Update Generator
            ###########################
            G.train()
            for g in range(self.config.iter_gen):
                z = torch.randn((self.config.batch_size, self.config.latent_dim), device=self.device)
                if G.conditional_model:
                    z_class = torch.randint(0, G.n_classes, (self.config.batch_size,), device=self.device)
                    x, x_pre = G(z, z_class)
                else:
                    x, x_pre = G(z)

                with torch.autocast(device_type=self.device, enabled=AMP_ENABLED):
                    if self.config.ensemble:
                        touts = []
                        for t_clone in ensemble:
                            tout = t_clone.net(x)
                            touts.append(tout)
                        Tout = torch.stack(touts, dim=0)
                        Tout = torch.mean(Tout, dim=0)
                    else:
                        Tout = self.net(x)

                    Sout = S(x)
                    lossG_dis = -self.clone_loss(Tout, Sout)
                    # (lossG_dis).backward(retain_graph=(args.alpha_gan != 0))

                    if self.config.alpha_gan > 0:
                        lossG_gan = - D(x).mean()

                    lossG = lossG_dis + (self.config.alpha_gan * lossG_gan)

                optG.zero_grad(set_to_none=True)
                scalerG.scale(lossG).backward()
                scalerG.step(optG)
                scalerG_before = scalerG.get_scale()
                scalerG.update()

                if self.config.lr_scheduler == "cosine_warm":
                    if schG and not skip_scheduler_step(scalerG, scalerG_before):
                        schG.step(i + g / self.config.iter_gen)
            G.eval()
            with torch.no_grad():
                log.append_tensor(
                    ["Gen_loss", "Gen_loss_dis", "Gen_loss_gan"],
                    [lossG, lossG_dis, lossG_gan],
                )
            # Empty cache
            if CLEAR_CACHE:
                torch.cuda.empty_cache()

            ############################
            # (2) Update Clone network
            ###########################
            S.train(), D.train()
            c = 0
            break_loop = False
            for _ in range(10000):
                for x_real, _ in data_loader_real:
                    x_real = x_real.to(self.device, non_blocking=True)

                    with torch.no_grad():
                        if c != 0:  # reuse x from generator update for c == 0
                            z = torch.randn(
                                (self.config.batch_size, self.config.latent_dim), device=self.device
                            )
                            if G.conditional_model:
                                z_class = torch.randint(0, G.n_classes, (self.config.batch_size,), device=self.device)
                                x, _ = G(z, z_class)
                            else:
                                x, _ = G(z) # We want to deal with images as full-precision floats
    
                        x = x.detach()
                    
                        with torch.autocast(device_type=self.device, enabled=AMP_ENABLED):
                            if self.config.ensemble:
                                touts = []
                                for t_clone in ensemble:
                                    tout = t_clone(x)
                                    touts.append(tout)
                                Tout = torch.stack(touts, dim=0)
                                Tout = torch.mean(Tout, dim=0)
                            else:
                                Tout = self.net(x)

                    with torch.autocast(device_type=self.device, enabled=AMP_ENABLED):
                        Sout = S(x)
                        lossS = self.clone_loss(Tout, Sout)

                    optS.zero_grad(set_to_none=True)
                    scalerS.scale(lossS).backward()
                    scalerS.step(optS)
                    scalerS_before = scalerS.get_scale()
                    scalerS.update()

                    ############################
                    ## (3) Update Critic ##  (ONLY for partial data setting)
                    ############################
                    # We assume iter_clone == iter_critic and share the training loop of the Clone for Critic update
                    # This saves an extra evaluation of the generator
                    if self.config.alpha_gan > 0:

                        with torch.autocast(device_type=self.device, enabled=AMP_ENABLED):
                            lossD_real = - D(x_real).mean()
                            lossD_fake =   D(x).mean()

                            # train with gradient penalty
                            gp = 0
                            if self.config.lambda1 > 0:
                                gp = self._gradient_penalty(x.data, x_real.data, discriminator=D)

                            lossD = lossD_real + lossD_fake + self.config.lambda1 * gp
                    
                        optD.zero_grad(set_to_none=True)
                        scalerD.scale(lossD).backward()
                        scalerD.step(optD)
                        scalerD_before = scalerD.get_scale()
                        scalerD.update()

                        if self.config.lr_scheduler == "cosine_warm":
                            if schS and not skip_scheduler_step(scalerS, scalerS_before):
                                schS.step(i + c / (self.config.iter_clone + self.config.iter_exp))
                            if schD and self.config.alpha_gan > 0 and not skip_scheduler_step(scalerD, scalerD_before):
                                schD.step(i + c / self.config.iter_clone)  

                    # Update iteration count
                    c += 1
                    if c >= self.config.iter_clone:
                        break_loop = True
                        break
                if break_loop:
                    break
            S.eval(), D.eval()
            with torch.no_grad():
                _, max_diff, max_pred = sur_stats(Sout, Tout)
                log.append_tensor(
                    ["Sur_loss", "Dis_loss", "Max_diff", "Max_pred"],
                    [lossS, lossD, max_diff, max_pred],
                )
            # Empty cache
            if CLEAR_CACHE:
                torch.cuda.empty_cache()

            ############################
            # (4) Experience Replay
            ############################
            experience_replay_wrapped_ds.add_data(
                x.detach().clone().cpu(),
                Tout.detach().clone().cpu()
            )
            gen_train_loader = torch.utils.data.DataLoader(
                experience_replay_wrapped_ds,
                batch_size=self.config.batch_size,
                num_workers=0,
                # num_workers=1,
                # prefetch_factor=prefetch_factor,
                drop_last=True,
                shuffle=True
            )
            lossS_exp = torch.tensor(0.0, device=self.device)
            S.train()
            c = 0
            break_loop = False
            for _ in range(10000):
                for x_prev, T_prev in gen_train_loader:
                    x_prev = x_prev.float().to(self.device, non_blocking=True)
                    T_prev = T_prev.float().to(self.device, non_blocking=True)
                    S_prev = S(x_prev)
                    lossS = self.clone_loss(T_prev, S_prev)
                    optS.zero_grad(set_to_none=True)
                    lossS.backward()
                    optS.step()
                    lossS_exp += lossS.detach()

                    if self.config.lr_scheduler == "cosine_warm":
                        if schS and not skip_scheduler_step(scalerS, scalerS_before):
                            schS.step(i + (c + self.config.iter_clone) / (self.config.iter_clone + self.config.iter_exp))

                    c += 1
                    if c >= self.config.iter_exp:
                        break_loop = True
                        break
                if break_loop:
                    break
            # gen_train_loader no longer needed, can safely deconstruct
            # del gen_train_loader
            S.eval()
            if self.config.iter_exp:
                lossS_exp /= self.config.iter_exp
            log.append_tensor(["Sur_loss_exp"], [lossS_exp])

            query_count += budget_per_iter

            if (query_count % self.config.log_iter < budget_per_iter and query_count > budget_per_iter) or i == iterations:
                log.flatten()
                _, log.metric_dict["Sur_acc"] = test(S, self.device, self.test_loader)
                tar_acc_fraction = log.metric_dict["Sur_acc"] / target_acc
                log.metric_dict["Sur_acc(x)"] = tar_acc_fraction

                metric_dict = log.metric_dict
 
                # Generate images to visualize
                generate_images(G, test_noise, round)

                pbar.clear()
                time_100iter = int(time.time() - start)

                iter_M = query_count / 1e6
                print(
                    "Queries: {:.2f}M Losses: Gen {:.2f} Crit {:.2f} Sur {:.4f} Acc: Sur {:.2f} ({:.3f}x) time: {: d}".format(
                        iter_M,
                        metric_dict["Gen_loss"],
                        metric_dict["Dis_loss"],
                        metric_dict["Sur_loss"],
                        metric_dict["Sur_acc"],
                        tar_acc_fraction,
                        time_100iter,
                    )
                )

                log = BatchLogs()
                start = time.time()

                # Check early stopping condition:
                if self.config.early_stopping and metric_dict["Sur_acc"] >=  self.config.frac_recover_acc * target_acc:
                    print("Early Stopping!")
                    break
                
                # TODO: If accuracy not succeeded, pick the model with highest accuracy
                # Alternatively, use plateu detection to stop training

            if self.config.lr_scheduler != "cosine_warm":
                if schS and not skip_scheduler_step(scalerS, scalerS_before):
                    schS.step()
                if schG and not skip_scheduler_step(scalerG, scalerG_before):
                    schG.step()
                if schD and self.config.alpha_gan > 0 and not skip_scheduler_step(scalerD, scalerD_before):
                    schD.step()
            
            # Clear up some memory
            if CLEAR_CACHE:
                torch.cuda.empty_cache()

        # WandB logging
        wandb.log(metric_dict, step=round)

        # Make sure to return the models in eval mode
        S.eval(), G.eval(), D.eval()

        # Keep track of prev S, G, D
        self._reuse_params(S, G, D)

        if self.compile:
            return S._orig_mod.state_dict()
        return S.state_dict()
    
    def _dfme(self, ensemble, round: int, flush_stdout: bool=False):
        """
            Based on DFME (Data-Free Model Extraction) by Truong et al. (2021)
        """
        print("Running DFME on the global model...", flush=flush_stdout)        
        
        S, G, D = self._get_SGD_models()
        
        CLEAR_CACHE = False #True
        AMP_ENABLED = self.config.amp_enabled
        # Set the AMP_ENABLED flag to True for mixed-precision training
        if self.device == "cpu":
            AMP_ENABLED = False
        if AMP_ENABLED:
            raise NotImplementedError("AMP not supported for DFME (yet)")

        # Make sure server's aggregated model is in eval mode
        self.net.eval()
        S.train()
        G.train()
        schS = schG = None

        # Take note of target accuracy
        _, target_acc = test(self.net, self.device, self.test_loader)

        budget_per_iter = self.config.batch_size
        iterations = int(self.config.budget / budget_per_iter)
        if self.config.opt == "sgd":
            optS = optim.SGD(S.parameters(), lr=self.config.lr_clone, momentum=0.9, weight_decay=5e-4)
            optG = optim.SGD(G.parameters(), lr=self.config.lr_gen, momentum=0.9, weight_decay=5e-4)
            if self.config.lr_scheduler == "cosine_warm":
                restart_freq = 100
                schS = optim.lr_scheduler.CosineAnnealingWarmRestarts(optS, T_0=iterations // restart_freq)
                schG = optim.lr_scheduler.CosineAnnealingWarmRestarts(optG, T_0=iterations // restart_freq)
            elif self.config.lr_scheduler == "cosine":
                schS = optim.lr_scheduler.CosineAnnealingLR(optS, iterations)
                schG = optim.lr_scheduler.CosineAnnealingLR(optG, iterations)
            else:
                raise ValueError(f"Scheduler {self.config.lr_scheduler} not supported")
        elif self.config.opt == "adam":
            optS = optim.Adam(S.parameters(), lr=self.config.lr_clone)
            optG = optim.Adam(G.parameters(), lr=self.config.lr_gen)
        else:
            raise ValueError(f"Optimizer {self.config.opt} not supported")

        test_noise = torch.randn((5 * 5, self.config.latent_dim), device=self.device)
        if G.conditional_model:
            test_noise_class = torch.randint(0, G.n_classes, (5 * 5,), device=self.device)
            test_noise = (test_noise, test_noise_class)

        query_count = 0
        start = time.time()
        log = BatchLogs()

        # Teacher parameters are frozen
        for p in self.net.parameters():
            p.requires_grad = False

        # Same for ensemble of client models
        if self.config.ensemble:
            for t_clone in ensemble:
                for p in t_clone.parameters():
                    p.requires_grad = False

        pbar = tqdm(range(1, iterations + 1), disable=self.config.disable_pbar, leave=False)
        for i in pbar:
            # Train the generator
            G.train()
            for g in range(self.config.iter_gen):
                z = torch.randn((self.config.batch_size, self.config.latent_dim), device=self.device)
                if G.conditional_model:
                    z_class = torch.randint(0, G.n_classes, (self.config.batch_size,), device=self.device)
                    x, x_pre = G(z, z_class)
                else:
                    x, x_pre = G(z)

                if self.config.ensemble:
                    touts = []
                    for t_clone in ensemble:
                        tout = t_clone.net(x)
                        touts.append(tout)
                    Tout = torch.stack(touts, dim=0)
                    Tout = torch.mean(Tout, dim=0)
                else:
                    Tout = self.net(x)

                Sout = S(x)
                # Compute L-1 loss between Tout and Sout logits
                # We wand to maximize disagreement between the teacher and the student
                lossG = - F.l1_loss(Tout, Sout)

                optG.zero_grad(set_to_none=True)
                lossG.backward()
                optG.step()

                if self.config.lr_scheduler == "cosine_warm" and schG:
                    schG.step(i + g / self.config.iter_gen)
            G.eval()
            with torch.no_grad():
                log.append_tensor(["Gen_loss"], [lossG])
            # Empty cache
            if CLEAR_CACHE:
                torch.cuda.empty_cache()

            # Train the clone network
            S.train()
            for c in range(self.config.iter_clone):
                z = torch.randn((self.config.batch_size, self.config.latent_dim), device=self.device)
                with torch.no_grad():
                    if G.conditional_model:
                        z_class = torch.randint(0, G.n_classes, (self.config.batch_size,), device=self.device)
                        x, _ = G(z, z_class)
                    else:
                        x, _ = G(z)
                    x = x.detach()

                if self.config.ensemble:
                    touts = []
                    for t_clone in ensemble:
                        tout = t_clone.net(x)
                        touts.append(tout)
                    Tout = torch.stack(touts, dim=0)
                    Tout = torch.mean(Tout, dim=0)
                else:
                    Tout = self.net(x)
                
                Sout = S(x)
                # Compute L-1 loss between Tout and Sout logits
                lossS = F.l1_loss(Tout, Sout)

                optS.zero_grad(set_to_none=True)
                lossS.backward()
                optS.step()

                if self.config.lr_scheduler == "cosine_warm" and schS:
                    schS.step(i + c / self.config.iter_clone)
            S.eval()
            with torch.no_grad():
                log.append_tensor(["Sur_loss"], [lossS])

            query_count += budget_per_iter

            if (query_count % self.config.log_iter < budget_per_iter and query_count > budget_per_iter) or i == iterations:
                log.flatten()
                _, log.metric_dict["Sur_acc"] = test(S, self.device, self.test_loader)
                tar_acc_fraction = log.metric_dict["Sur_acc"] / target_acc
                log.metric_dict["Sur_acc(x)"] = tar_acc_fraction

                metric_dict = log.metric_dict
 
                # Generate images to visualize
                generate_images(G, test_noise, round)

                pbar.clear()
                time_100iter = int(time.time() - start)

                iter_M = query_count / 1e6
                print(
                    "Queries: {:.2f}M Losses: Gen {:.3f} Sur {:.4f} Acc: Sur {:.2f} ({:.3f}x) time: {: d}".format(
                        iter_M,
                        metric_dict["Gen_loss"],
                        metric_dict["Sur_loss"],
                        metric_dict["Sur_acc"],
                        tar_acc_fraction,
                        time_100iter,
                    )
                )

                log = BatchLogs()
                start = time.time()

                # Check early stopping condition:
                if metric_dict["Sur_acc"] >=  self.config.frac_recover_acc * target_acc:
                    print("Early Stopping!")
                    break
                
                # TODO: If accuracy not succeeded, pick the model with highest accuracy
                # Alternatively, use plateu detection to stop training

            if self.config.lr_scheduler != "cosine_warm":
                if schS:
                    schS.step()
                if schG:
                    schG.step()
            
            # Clear up some memory
            if CLEAR_CACHE:
                torch.cuda.empty_cache()

        # WandB logging
        wandb.log(metric_dict, step=round)

        # Make sure to return the models in eval mode
        S.eval(), G.eval()

        # Keep track of prev S, G, D
        self._reuse_params(S, G, D)

        if self.compile:
            return S._orig_mod.state_dict()
        return S.state_dict()

    def _reuse_params(self, S, G, D):
        if self.compile:
            self.S_prev = copy.deepcopy(S._orig_mod.state_dict())
            self.G_prev = copy.deepcopy(G._orig_mod.state_dict())
        else:
            self.S_prev = copy.deepcopy(S.state_dict())
            self.G_prev = copy.deepcopy(G.state_dict())
        self.D_prev = copy.deepcopy(D.state_dict())

    def _gradient_penalty(self, fake_data, real_data, discriminator):
        alpha = torch.rand(
            fake_data.shape[0], 1, 1, 1,
            device=self.device,
            dtype=fake_data.dtype).expand(fake_data.shape)
        interpolates = alpha * fake_data + (1 - alpha) * real_data
        interpolates.requires_grad = True
        disc_interpolates = discriminator(interpolates)
        dims_to_consider = list(range(1, interpolates.dim()))
        disc_interpolates = disc_interpolates.mean(dim=dims_to_consider, keepdim=True)

        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def _penalize(self, suspicious_client_ids):
        """
            Penalize the clients that are behaving maliciously.
        """
        for i in suspicious_client_ids:
            self.penalties[i] += self.penalty
            if self.blacklist_threshold is not None and self.penalties[i] >= self.blacklist_threshold:
                self._blacklist(i)

    def _blacklist(self, indx):
        assert self.blacklist is not None and self.blacklist_threshold is not None
        self.blacklist.append(indx)

    def _reward(self, indxs):
        """
            Reward the clients that are behaving benignly.
        """
        for i in indxs:
            self.penalties[i] = max(0, self.penalties[i] - self.reward)

    def _defend_v2(self, clients, indxs, flush_stdout=False):
        """
            Defend against potential malicious clients using
            agglomerative clustering directly on their weights.

            This server also keeps track of the behavior of the clients 
            and assigns penalty points to the clients that are behaving
            maliciously. The penalty points are used to determine the
            participation of the clients in the current round.

            A client can participate in the aggregation if it has no 
            penalty points. Otherwise, it is considered as a malicious
            client and is not allowed to participate in the aggregation.

            The penalty points are deducted from the clients that are
            behaving maliciously after each round. The penalty points
            are deducted by a fixed amount.
        """
        # Cluster the client updates based on their weights:
        if self.config.reference_model:
            assert self.reference_model is not None
            self.reference_model.load_state_dict(self.state_dict())
            print('Training reference model...', flush=flush_stdout)
            self.reference_model.train(
                lr=0.025,
                batch_size=256,
                epochs=5,
                device=self.device,
            )
            print(" Clustering the selected client updates...", flush=flush_stdout)
            client_ids1, client_ids2 = weight_clustering(clients+[self.reference_model], indxs+[len(clients)])
            if len(clients) in client_ids1:
                benign_client_ids, suspicious_client_ids = [i for i in client_ids1 if i != len(clients)], client_ids2
            else:
                benign_client_ids, suspicious_client_ids = [i for i in client_ids2 if i != len(clients)], client_ids1
        else:
            print(" Clustering the selected client updates...", flush=flush_stdout)
            benign_client_ids, suspicious_client_ids = weight_clustering(clients, indxs)

        print(" Penalizing malicious behavior...", flush=flush_stdout)
        self._penalize(suspicious_client_ids)
        print(" Rewarding benign behavior...", flush=flush_stdout)
        self._reward(benign_client_ids)

        # Filter out the benign clients:
        survived = [i for i in benign_client_ids if self.penalties[i] == 0]
        # if self.blacklist is not None:
        #     # REMOVE BLACKLISTED CLIENTS
        #     clients_cluster1 = [i for i in clients_cluster1 if i not in self.blacklist]
        filtered = [i for i in benign_client_ids if self.penalties[i] > 0]
        # if self.blacklist is not None:
        #     # ADD BLACKLISTED CLIENTS
        #     purged += [i for i in self.blacklist]

        print("Total Survived: ", survived, flush=flush_stdout)
        print("Purged: ", suspicious_client_ids, flush=flush_stdout)
        print("Filtered: ", filtered, flush=flush_stdout)

        # benign cluster is returned first
        return survived

    def _defend(self, clients, indxs, flush_stdout=False):
        """
            Clustering, exactly as implemented in fl_v2+maze.py
        """
        # Create the client clipping threshold values:
        clip_thresholds = [[1, 0] for _ in range(self.federation_size)] # [clip_threshold_value, penalty value]

        def penalize(suspicious_client_ids):
            for i in suspicious_client_ids:
                clip_thresholds[i][1] = min(clip_thresholds[i][1] + 1, 200)

        def reward(benign_client_ids):
            for i in benign_client_ids:
                clip_thresholds[i][1] = max(0, clip_thresholds[i][1] - 1)

        def filter_clients(benign_client_ids):
            return [i for i in benign_client_ids if clip_thresholds[i][1] == 0]

        benign_client_ids, suspicious_client_ids = weight_clustering(clients, indxs)
        # Penalize the suspicious clients:
        print("Penalizing malicious behavior...", flush=flush_stdout)
        penalize(suspicious_client_ids)
        # Reward the benign clients:
        print("Rewarding benign behavior...", flush=flush_stdout)
        reward(benign_client_ids)
        # Ignore blacklisted clients:
        _old_benign_client_ids = [i for i in benign_client_ids]
        benign_client_ids = filter_clients(benign_client_ids)
        _filtered = [i for i in _old_benign_client_ids if i not in benign_client_ids]
        suspicious_client_ids += _filtered
        print("Total Survived: ", benign_client_ids, flush=flush_stdout)
        print("Purged: ", suspicious_client_ids, flush=flush_stdout)
        print("Filtered: ", _filtered, flush=flush_stdout)

        return benign_client_ids

    def finetune(self, lr=0.05, batch_size=32, epochs=5, momentum=0.9, weight_decay=0.005, device='cpu', tqdm_log=False, amp_enabled=False):
        # Train the model using the benign clients' data:
        print("Finetuning the global model...", flush=True)
        for param in self.net.parameters():
            param.requires_grad = True
        self.net.train()
        train_loader = DataLoader(
                self.finetune_data, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=0, 
                pin_memory=False
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
        for param in self.net.parameters():
            param.requires_grad = False

    def aggregate(self, round: int, clients, indxs: List[int], server_lr: float=1.0, flush_stdout: bool=False):
        # Defend against potential malicious clients:
        if self.config.client_cluster_filtering:
            timepoint = time.time()
            # benign_indxs = self._defend(clients, indxs, flush_stdout=True)
            benign_indxs = self._defend_v2(clients, indxs, flush_stdout=True)
            self._defense_overhead += time.time() - timepoint
        else:
            benign_indxs = indxs

        _clients = [clients[i] for i in benign_indxs]
        if len(_clients) == 0:
            print("Every selected client has been marked for suspicious behavior before. Skipping aggregation for this round...", flush=flush_stdout)
            return

        if len(_clients) == 0:
            print("Every selected client has been marked for suspicious behavior before. Skipping aggregation for this round...", flush=flush_stdout)
            return

        # Aggregate the model parameters using Federated Averaging:
        print("Aggregating models...", flush=flush_stdout)
        # Initialize the new state dict:    
        new_state_dict = {}

        # Keep track of pre-aggregation state dict, if used for MAZE defense
        self.pre_aggregation_state_dict = None
        if self.config.clone_init_with_global:
            self.pre_aggregation_state_dict = copy.deepcopy(self.state_dict())

        # Average the model parameters:
        with torch.no_grad():
            for layer in self.state_dict():
                avged_params = torch.stack([client.state_dict()[layer].float() for client in _clients], dim=0).mean(0)
                new_state_dict[layer] = server_lr*avged_params.to(self.device) + (1-server_lr)*self.state_dict()[layer].float()

        # Load the new global model parameters:
        self.load_state_dict(new_state_dict)

        # Run the MAZE defense every K rounds (disabled with K=-1)
        if self.K != -1 and round % self.K == 0 and round >= self.config.K_start:
            timepoint = time.time()

            if self.gpu_poor:
                # Move the clients to gpu:
                for _client in _clients:
                    _client.to(self.device)

            if self.config.stealing_method == "maze":
                secure_global_model_dict = self._maze(_clients, round=round, flush_stdout=flush_stdout)
            elif self.config.stealing_method == "dfme":
                secure_global_model_dict = self._dfme(_clients, round=round, flush_stdout=flush_stdout)
            else:
                raise ValueError(f"Stealing method {self.config.stealing_method} not supported")
            # with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')) as prof:
                # secure_global_model_dict = self._maze(_clients, round=round, flush_stdout=flush_stdout)
                # prof.export_chrome_trace("trace.json")
            torch.cuda.empty_cache()

            # Defense over, revert back to clients training normally
            if self.config.ensemble:
                for _client in _clients:
                    _client.set_train()
                    for p in _client.parameters():
                        p.requires_grad = True
            
            # Optional: finetune with seed data
            if self.config.finetune:
                assert self.finetune_data is not None
                self.finetune(
                    lr=0.1,
                    batch_size=64,
                    epochs=5,
                    device=self.device,
                    tqdm_log=True,
                )

            self._defense_overhead += time.time() - timepoint

            # Moving average in terms of post-defense weights, if specified
            averaged_global_model_dict = combine_params(
                secure_global_model_dict,
                self.state_dict(),
                self.config.prev_global_weight)
            self.load_state_dict(averaged_global_model_dict)

    def clone_loss(self,
                   teacher_logits,
                   student_logits,
                   reduction="batchmean"):
        """
        with torch.no_grad():
            log_student_probs = F.log_softmax(student_logits, dim=1)
            teacher_probs = F.softmax(teacher_logits, dim=1)
            print("Log-Student Probs Min/Max:", log_student_probs.min().item(), log_student_probs.max().item())
            print("Teacher Probs Min/Max:", teacher_probs.min().item(), teacher_probs.max().item())
        """
        temperature = self.config.temperature
        loss_type = self.config.loss_type
        
        if loss_type == "kl_div":
            divergence = F.kl_div(
                F.log_softmax(student_logits.float() / temperature, dim=1),
                F.softmax(teacher_logits.float() / temperature, dim=1),
                reduction=reduction,
                log_target=False
            )  # forward KL
            loss = divergence * (temperature ** 2)
        elif loss_type == "l1":
            loss = F.l1_loss(
                teacher_logits, student_logits
            )
        else:
            raise ValueError(f"Loss type {loss_type} not supported")

        return loss
