"""
Configuration Management for Simple Cluster Router

Handles configuration loading from environment variables, files, and command line arguments.
Provides secure management of API keys and other sensitive information.
"""
import os
import json
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import logging


@dataclass
class SimpleClusterConfig:
    """
    Configuration class for Simple Cluster Router.
    
    Attributes:
        data_path (str): Path to input JSONL file
        train_ratio (float): Ratio of data used for training (0.0-1.0)
        seed (int): Random seed for reproducible results
        n_clusters (int): Number of K-means clusters to create
        max_router (int): Number of top models to select for routing
        top_k (int): Number of closest clusters to consider for routing
        beta (float): Temperature parameter for cluster probability (higher = more focused)
        max_workers (int): Number of concurrent workers for embedding generation
        cluster_batch_size (int): Batch size for cluster processing
        max_tokens (int): Maximum tokens per query for embedding (conservative limit)
        embedding_model (str): Name of embedding model to use
        embedding_base_url (str): Base URL for embedding API
        embedding_api_key (str): API key for embedding service
    """
    
    # Required parameters
    data_path: str
    
    # Core algorithm parameters
    train_ratio: float = 0.7
    seed: int = 42
    n_clusters: int = 32
    max_router: int = 1
    top_k: int = 1
    beta: float = 9.0
    
    # Performance parameters
    max_workers: int = 4
    cluster_batch_size: int = 1000
    max_tokens: int = 7500  # Conservative limit to avoid API errors
    
    # Embedding service configuration
    embedding_model: str = "text-embedding-3-large"
    embedding_base_url: str = "http://172.30.5.197:8000/v1"
    embedding_api_key: str = "inplaceholder"
    
    # Balance router parameters
    cost_sensitivity: float = 0.3  # Weight for cost consideration (0.0-1.0)
    performance_weight: float = 0.7  # Weight for performance consideration (0.0-1.0)
    min_accuracy_threshold: float = 0.0  # Minimum accuracy requirement (0.0-1.0)
    budget_limit: Optional[float] = None  # Maximum cost per query (None = no limit)
    
    # Model filtering parameters
    excluded_models: List[str] = field(default_factory=list)  # Models to exclude from routing
    
    # Dataset filtering parameters
    excluded_datasets: List[str] = field(default_factory=list)  # Datasets to exclude from evaluation
    dataset_exclusion_mode: str = "hard"  # "soft" or "hard" - soft: exclude from eval but include in clustering, hard: exclude completely
    
    # Model export parameters
    export_cluster: Optional[str] = None  # Path to export trained cluster models (normalizer, centers, rankings)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        
    def _validate_config(self):
        """Validate configuration parameters."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError(f"train_ratio must be between 0 and 1, got {self.train_ratio}")
            
        if self.n_clusters <= 0:
            raise ValueError(f"n_clusters must be positive, got {self.n_clusters}")
            
        if self.max_router <= 0:
            raise ValueError(f"max_router must be positive, got {self.max_router}")
            
        if not self.embedding_api_key:
            raise ValueError("EMBEDDING_API_KEY environment variable is required")
            
        if self.max_tokens > 8000:
            logging.warning(f"max_tokens={self.max_tokens} may exceed API limits, consider reducing to <8000")
            
        # Validate balance router parameters
        if not 0.0 <= self.cost_sensitivity <= 1.0:
            raise ValueError(f"cost_sensitivity must be between 0.0 and 1.0, got {self.cost_sensitivity}")
            
        if not 0.0 <= self.performance_weight <= 1.0:
            raise ValueError(f"performance_weight must be between 0.0 and 1.0, got {self.performance_weight}")
            
        if not 0.0 <= self.min_accuracy_threshold <= 1.0:
            raise ValueError(f"min_accuracy_threshold must be between 0.0 and 1.0, got {self.min_accuracy_threshold}")
            
        if self.budget_limit is not None and self.budget_limit <= 0:
            raise ValueError(f"budget_limit must be positive, got {self.budget_limit}")
        
        # Validate excluded_models
        if self.excluded_models and not isinstance(self.excluded_models, list):
            raise ValueError(f"excluded_models must be a list, got {type(self.excluded_models)}")
        
        # Validate excluded_datasets
        if self.excluded_datasets and not isinstance(self.excluded_datasets, list):
            raise ValueError(f"excluded_datasets must be a list, got {type(self.excluded_datasets)}")
        
        # Validate dataset_exclusion_mode
        if self.dataset_exclusion_mode not in ["soft", "hard"]:
            raise ValueError(f"dataset_exclusion_mode must be 'soft' or 'hard', got '{self.dataset_exclusion_mode}'")
        
        # Validate that performance_weight and cost_sensitivity sum to reasonable range
        total_weight = self.performance_weight + self.cost_sensitivity
        if total_weight <= 0:
            raise ValueError("Sum of performance_weight and cost_sensitivity must be positive")
        if total_weight > 2.0:
            logging.warning(f"Sum of weights ({total_weight:.2f}) is greater than 2.0, consider normalizing")
        
        # Validate export_cluster path
        if self.export_cluster is not None:
            export_path = Path(self.export_cluster)
            if not export_path.parent.exists():
                raise ValueError(f"Export cluster directory does not exist: {export_path.parent}")

    @classmethod
    def from_env(cls, data_path: str, **kwargs) -> 'SimpleClusterConfig':
        """
        Create configuration from environment variables.
        
        Args:
            data_path: Path to input data file
            **kwargs: Override default configuration values
            
        Returns:
            SimpleClusterConfig instance
        """
        config_dict = {"data_path": data_path}
        
        # Load from environment variables
        env_mappings = {
            "TRAIN_RATIO": ("train_ratio", float),
            "SEED": ("seed", int),
            "N_CLUSTERS": ("n_clusters", int),
            "MAX_ROUTER": ("max_router", int),
            "TOP_K": ("top_k", int),
            "BETA": ("beta", float),
            "MAX_WORKERS": ("max_workers", int),
            "CLUSTER_BATCH_SIZE": ("cluster_batch_size", int),
            "MAX_TOKENS": ("max_tokens", int),
            "EMBEDDING_MODEL": ("embedding_model", str),
            "EXCLUDED_MODELS": ("excluded_models", str),  # Comma-separated string
            "EXCLUDED_DATASETS": ("excluded_datasets", str),  # Comma-separated string
            "DATASET_EXCLUSION_MODE": ("dataset_exclusion_mode", str),
        }
        
        for env_var, (attr_name, attr_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if attr_name in ["excluded_models", "excluded_datasets"]:
                        # Parse comma-separated string to list
                        config_dict[attr_name] = [item.strip() for item in env_value.split(",") if item.strip()]
                    else:
                        config_dict[attr_name] = attr_type(env_value)
                except ValueError as e:
                    logging.warning(f"Invalid value for {env_var}={env_value}: {e}")
        
        # Override with provided kwargs
        config_dict.update(kwargs)
        
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_file: str, data_path: str) -> 'SimpleClusterConfig':
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration JSON file
            data_path: Path to input data file
            
        Returns:
            SimpleClusterConfig instance
        """
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config_dict["data_path"] = data_path
        return cls(**config_dict)
    
    def save(self, config_file: str):
        """
        Save configuration to JSON file (excluding sensitive information).
        
        Args:
            config_file: Path to save configuration
        """
        # Create dict excluding sensitive information
        safe_config = {}
        for key, value in self.__dict__.items():
            if 'key' not in key.lower() and 'secret' not in key.lower():
                safe_config[key] = value
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(safe_config, f, indent=2, ensure_ascii=False)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary (excluding sensitive info)."""
        return {
            'data_path': self.data_path,
            'train_ratio': self.train_ratio,
            'seed': self.seed,
            'n_clusters': self.n_clusters,
            'max_router': self.max_router,
            'top_k': self.top_k,
            'beta': self.beta,
            'max_workers': self.max_workers,
            'cluster_batch_size': self.cluster_batch_size,
            'max_tokens': self.max_tokens,
            'embedding_model': self.embedding_model,
            'cost_sensitivity': self.cost_sensitivity,
            'performance_weight': self.performance_weight,
            'min_accuracy_threshold': self.min_accuracy_threshold,
            'budget_limit': self.budget_limit,
            'excluded_models': self.excluded_models,
            'excluded_datasets': self.excluded_datasets,
            'dataset_exclusion_mode': self.dataset_exclusion_mode,
            'export_cluster': self.export_cluster,
        }


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('cluster_router.log', encoding='utf-8')
        ]
    )