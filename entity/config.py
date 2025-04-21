
"""
    Definitions for configurations.
"""
from dataclasses import dataclass, field
from typing import Optional, List
from simple_parsing.helpers import Serializable


@dataclass
class FederationConfig(Serializable):
    """
        Configuration for the federation.
    """
    size: int
    """Number of clients in the federation."""
    number_of_malicious_clients: Optional[int] = 0
    """Number of malicious clients in the federation."""
    round_size: Optional[int] = 10
    """Number of clients in each round."""
    rounds: Optional[int] = 100
    """Number of rounds for the federated learning."""
    malicious_client_strategy: Optional[str] = "random"
    """Strategy for malicious client selection per round. Should be one of ["random", "enforced"]"""


@dataclass
class SettingsConfig(Serializable):
    """
        Configuration for the environment.
    """
    seed: int = 42
    """Random seed for reproducibility."""
    cuda: int = 0
    """CUDA device."""
    verbose: bool = True
    """Verbose mode."""
    flush_print: bool = True
    """Flush print."""
    parallel_client_training: bool = False
    """Train clients in parallel."""
    num_parallel_max: int = 16
    """Maximum number of parallel clients."""


@dataclass
class AttackConfig(Serializable):
    """
        Configuration for the poisoning attack.
    """
    type: str
    """Type of the attack."""
    DPR: float
    """Data poisoning ratio."""
    target_class: int
    """Target class for the attack."""
    trigger_pattern: str
    """Trigger pattern for the attack."""
    trigger_position: str
    """Trigger position for the attack."""
    victim_class: Optional[int] = None
    """Victim class for the attack."""
    non_iid_equal_distribution: Optional[bool] = False
    """For non-IID setting, accumulate all poison data for bad clients and distribute it evenly?"""
    participation_strategy: Optional[str] = "consistent"
    """Participation strategy for malicious clients. Defaults to consistent."""
    poisoning_start_round: Optional[int] = None
    """When to start poisoning data as a malicious adversary. Defaults to 0 (start from the beginning)."""
    poisoning_end_round: Optional[int] = None
    """When to stop poisoning data as a malicious adversary. Defaults to None (continue poisoning)."""
    # Neurotoxin hyperparameters
    neurotoxin_top_k: Optional[float] = None
    """Top-k value to use if Neurotoxin attack is used. Should be in [0, 1] range."""
    # Chameleon hyperparemeters
    poisoned_projection_norm: Optional[float] = 5
    """Norm of the projection of the poisoned data. Defaults to 5."""
    poisoned_is_projection_grad: Optional[bool] = False
    """If true, the projection is done on the gradient of the poisoned data. Defaults to False."""
    poisoned_supcon_lr: Optional[float] = 0.015
    """Learning rate for the supervised contrastive loss. Defaults to 0.015."""
    poisoned_supcon_momentum: Optional[float] = 0.9
    """Momentum for the supervised contrastive loss. Defaults to 0.9."""
    poisoned_supcon_weight_decay: Optional[float] = 0.0005
    """Weight decay for the supervised contrastive loss. Defaults to 0.0005."""
    malicious_milestones: Optional[List[int]] = field(default_factory=lambda: [2, 4, 6, 8])
    """Milestones for the malicious clients. Defaults to [2, 4, 6, 8]."""
    malicious_lr_gamma: Optional[float] = 0.3
    """Learning rate gamma for the malicious clients. Defaults to 0.3."""
    fac_scale_weight: Optional[float] = 2
    """Scale weight for the factorization. Defaults to 2."""

    def __post_init__(self):
        if self.participation_strategy == "after_convergence" and self.poisoning_start_round is None:
            raise ValueError("poisoning_start_round must be specified when participation_strategy is after_convergence.")
        if self.neurotoxin_top_k is not None:
            if self.neurotoxin_top_k < 0 or self.neurotoxin_top_k > 1:
                raise ValueError("neurotoxin_top_k should be a real number in the range [0, 1].")


@dataclass
class ClientConfig(Serializable):
    """
        Configuration for the client.
    """
    IID: bool
    """Is client data IID?"""
    num_epochs: int
    """Number of epochs for local client training."""
    lr: float
    """Learning rate for local client training."""
    batch_size: int
    """Batch size for local client training."""
    momentum: Optional[float] = 0.9
    """Momentum for local client training."""
    weight_decay: Optional[float] = 0.005
    """Weight decay for local client training."""
    batchwise_poison: Optional[bool] = False
    """Inject poison in each batch of data?"""
    dirichlet_alpha: Optional[float] = 1.0
    """Dirichlet alpha, used if non-IID data."""
    ADR: Optional[float] = 1.0
    """How much of the original dataset should be used for experiments. Defaults to using all data"""
    transform: Optional[bool] = True
    """Use data transformations locally?"""
    amp_enabled: Optional[bool] = False
    """Use mixed-precision training"""
    amp_enabled_only_for_clean: Optional[bool] = False
    """If amp_enabled is True, use it only for benign clients"""
    attack: Optional[AttackConfig] = None
    """Attack configuration for the client (if all clients are not benign)"""

    def __post_init__(self):
        # Make sure ADR is in [0, 1] range
        if self.ADR < 0 or self.ADR > 1:
            raise ValueError("ADR must be in [0, 1] range.")


@dataclass
class DROPConfig(Serializable):
    """
        Configuration for the DROP defense.
    """
    K: int
    """How often to run the defense. Set to -1 to disable generative component"""
    budget: int
    """Number of queries"""
    model_gen: str
    """Model for the generator"""
    model_dis: str
    """Model for the discriminator"""
    stealing_method: Optional[str] = "maze"
    """What method to use for stealing"""
    client_cluster_filtering: Optional[bool] = True
    """Filter out potentially malicious clients with clustering"""
    ensemble: Optional[bool] = True
    """Use ensemble of models"""
    frac_recover_acc: Optional[float] = 1.0
    """Break when this fraction of the accuracy is recovered during distillation"""
    temperature: Optional[float] = 1.0
    """Temperature to use when dealing with logits in loss computation"""
    K_start: Optional[int] = 0
    """Start from iteration K_start"""
    num_repeat_real: Optional[int] = 10
    """Number of times to repeat real data"""
    prev_global_weight: Optional[float] = 0.0
    """When aggregating weights, use this weight for the previous global model"""
    random_weight_S: Optional[float] = 1.0
    """Weight to give to random-init in moving average of weights (for S)"""
    random_weight_G: Optional[float] = 1.0
    """Weight to give to random-init in moving average of weights (for G)"""
    random_weight_D: Optional[float] = 1.0
    """Weight to give to random-init in moving average of weights (for D)"""
    alpha_gan: Optional[float] = 0.5
    """Alpha for the GAN loss"""
    lambda1: Optional[float] = 10
    """Lambda for the L1 loss"""
    amp_enabled: Optional[bool] = False
    """Use mixed-precision training"""
    log_iter: Optional[int] = 1e5
    """Print stats every log_iter iterations"""
    disable_pbar: Optional[bool] = False
    """Disable progress bar"""
    iter_clone: Optional[int] = 5
    """Number of iterations for clone model"""
    iter_exp: Optional[int] = 10
    """Number of iterations for experience replay"""
    iter_gen: Optional[int] = 1
    """Number of iterations for generator"""
    lr_clone: Optional[float] = 0.1
    """Learning rate for clone model"""
    lr_gen: Optional[float] = 1e-3
    """Learning rate for generator"""
    lr_dis: Optional[float] = 1e-4
    """Learning rate for discriminator"""
    lr_scheduler: Optional[str] = "cosine"
    """Learning rate scheduler"""
    opt: Optional[str] = "sgd"
    """Optimizer"""
    latent_dim: Optional[int] = 100
    """Latent dimension for the generator"""
    blacklist_threshold: Optional[float] = None
    """Threshold for blacklisting"""
    batch_size: Optional[int] = 128
    """Batch size for the attack"""
    num_seed: Optional[int] = 100
    """Number of clean-data seeds for the attack"""
    penalty: Optional[float] = 1.0
    """Pelanty for clustering pre-defense (v2)"""
    reward: Optional[float] = None
    """Reward for clustering pre-defense (v2). If None, use penalty"""
    augment: Optional[bool] = False
    """Use augmentation when cycling through data?"""
    clone_init_with_global: Optional[bool] = False
    """Initialize the clone model with previous-round's global model weights?"""
    early_stopping: Optional[bool] = False
    """Use early stopping for DROP"""
    finetune: Optional[bool] = False
    """Finetune the clone model with the seed data post-extraction."""
    loss_type: Optional[str] = "kl_div"
    """Student-teacher loss to use in model-stealing in MAZE. Defaults to kl_div"""
    reference_model: Optional[bool] = False
    """Use the reference model for clustering"""
    random_S_alpha: Optional[float] = 0.0
    """???"""


@dataclass
class FLAREConfig(Serializable):
    """
        Configuration for the FLARE defense
    """
    temperature: Optional[float] = 1.0
    """Temperature for the softmax used for trust scores"""
    num_seed: Optional[int] = 100
    """Number of clean-data seeds for the attack"""
    batch_size: Optional[int] = 512
    """Batch size for computing PLRs"""

@dataclass
class MultiKrumConfig(Serializable):
    """
        Configuration for the Multi-Krum defense
    """
    multi_k: int


@dataclass
class FLIPConfig(Serializable):
    """
        Configuration for the FLIP defense
    """
    size_min: Optional[int] = 5
    """Consider a class as 'present' only when at least this many samples are present. Use 1 for EMNIST"""
    defense_start_round: Optional[int] = 0
    """Round after which defense starts"""
    attack_succ_threshold: Optional[float] = 0.85
    """CIFAR10 uses 0.85, MNIST uses 0.90"""
    trigger_size_check_threshold: Optional[float] = 0.25
    """Termination condition (fraction of pixels) for trigger size. 0.25 for CIFAR10, 0.125 for MNIST"""
    portion: Optional[float] = 0.9
    """0.9 for CIFAR10, 0.5 for MNIST"""
    trigger_steps: Optional[List[int]] = field(default_factory=lambda: [600, 600, 200])
    """Use [500, 400] for MNIST, [600, 600, 200] for CIFAR10"""
    init_mask_chunk: Optional[bool] = True
    """Init mask has some part of the image set to True. Use False for MNIST"""


@dataclass
class FLAMEConfig(Serializable):
    """
        Configuration for the FLAME defense
    """
    noise_sigma: float
    """Noise level"""


@dataclass
class FLTrustConfig(Serializable):
    """
        Configuration for the FLTrust defense
    """
    pass

@dataclass
class MedianConfig(Serializable):
    """
        Configuration for the Median defense
    """
    pass


@dataclass
class RandomAggregationConfig(Serializable):
    """
        Configuration for the RandomAggregation defense
    """
    softmax: Optional[bool] = False
    """Use softmax normalization for random weights"""
    granularity: Optional[str] = "client"
    """Granularity of aggregation"""
    top_k: Optional[int] = None
    """If not none, concentrate weights on top-k selections"""

@dataclass
class ServerV2Config(Serializable):
    """
        Configuration for the Server V2
    """
    pass

@dataclass
class FoolsGoldConfig(Serializable):

    """
        Configuration for the FoolsGold defense
    """
    pass


@dataclass
class DefenseConfig(Serializable):
    """
        Configuration for the defense.
    """
    type: str
    """Type of the defense."""
    defense_args: Optional[dict] = None
    """Arguments for the defense."""
    pretrain: Optional[bool] = False
    """If true, pretrain global model with auxiliary data"""

    def __post_init__(self):
        """
        Dynamically assign the specific defense config class based on the defense name.
        """
        if self.defense_args is not None and type(self.defense_args) is dict:
            wrapped_defense_config = self._resolve_defense_config(self.defense_args.copy())
            self.defense_args = wrapped_defense_config

    def _resolve_defense_config(self, dict_to_wrap: dict):
        """
        Resolve and cast to the appropriate defense configuration class based on the defense name.
        """
        if self.type.lower() == "maze":
            return DROPConfig(**dict_to_wrap)
        elif self.type.lower() == "flame":
            return FLAMEConfig(**dict_to_wrap)
        elif self.type.lower() == "fltrust":
            return FLTrustConfig(**dict_to_wrap)
        elif self.type.lower() == "multikrum":
            return MultiKrumConfig(**dict_to_wrap)
        elif self.type.lower() == "foolsgold":
            return FoolsGoldConfig(**dict_to_wrap)
        elif self.type.lower() == "median":
            return MedianConfig(**dict_to_wrap)
        elif self.type.lower() == "flare":
            return FLAREConfig(**dict_to_wrap)
        elif self.type.lower() == "random_aggregation":
            return RandomAggregationConfig(**dict_to_wrap)
        elif self.type.lower() == "flip":
            return FLIPConfig(**dict_to_wrap)
        elif self.type.lower() == "v2":
            return ServerV2Config(**dict_to_wrap)
        else:
            raise ValueError(f"Unknown defense name: {self.type}")


@dataclass
class ServerConfig(Serializable):
    """
        Configuration for the server.
    """
    server_lr: float
    """Learning rate for the server."""
    defense: DefenseConfig = None
    """Defense configuration for the server. None means undefended server"""
    low_confidence_threshold: Optional[float] = None
    """Low-confidence rejection threshold. If specified, any prediction by the server with confidence below this threshold is rejected."""
    gpu_poor: Optional[bool] = False
    """If true, shift around models to/from CPU to reduce peak memory usage"""
    def __post_init__(self):
        if self.low_confidence_threshold is not None:
            if self.low_confidence_threshold < 0 or self.low_confidence_threshold > 1:
                raise ValueError("low_confidence_threshold, when specified, must be in [0, 1] range.")


@dataclass
class ExperimentConfig(Serializable):
    """
        Experiment for inferring the retriever model.
    """
    dataset: str
    """Dataset for the experiment."""
    model: str
    """Model for the experiment."""
    client: ClientConfig
    """Client configuration for the experiment."""
    server: ServerConfig
    """Server configuration for the experiment."""
    federation: FederationConfig
    """Federation configuration for the experiment."""

    settings: Optional[SettingsConfig] = field(default_factory=SettingsConfig)
    """Settings for the experiment."""
