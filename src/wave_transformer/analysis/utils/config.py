"""
Configuration Management for Wave Transformer Analysis

Provides dataclass-based configuration with YAML support for analysis tools.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import warnings

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None


@dataclass
class CollectorConfig:
    """Configuration for WaveCollector"""
    enabled: bool = True
    sampling_rate: float = 1.0  # Fraction of batches to collect (0.0-1.0)
    max_samples_per_layer: int = 1000
    track_gradients: bool = False
    store_full_waves: bool = True
    compute_statistics: bool = True


@dataclass
class VisualizationConfig:
    """Configuration for visualization tools"""
    # TensorBoard settings
    use_tensorboard: bool = True
    tensorboard_log_dir: str = 'runs/wave_analysis'
    tb_flush_secs: int = 120

    # Weights & Biases settings
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)

    # General visualization settings
    plot_frequency: int = 1000  # Steps between visualizations
    save_figures: bool = True
    figure_format: str = 'png'  # 'png', 'pdf', 'svg'
    dpi: int = 150


@dataclass
class IntrospectionConfig:
    """Configuration for introspection tools"""
    # Harmonic analysis
    harmonic_importance_method: str = 'energy'  # 'energy', 'amplitude', 'variance'
    track_top_k_harmonics: int = 16

    # Spectrum tracking
    enable_spectrum_tracking: bool = True
    spectrum_snapshot_frequency: int = 5000

    # Interference analysis
    enable_interference_analysis: bool = False
    interference_window_size: int = 10

    # Layer analysis
    enable_layer_analysis: bool = True
    layer_snapshot_frequency: int = 1000


@dataclass
class TrainingConfig:
    """Configuration for training monitoring"""
    # Hook settings
    enable_hooks: bool = True
    hook_layers: List[str] = field(default_factory=lambda: ['all'])  # ['all'] or specific layer names

    # Gradient monitoring
    enable_gradient_monitoring: bool = True
    gradient_clip_threshold: Optional[float] = None

    # Callback settings
    enable_callbacks: bool = True
    checkpoint_frequency: int = 5000
    validation_frequency: int = 1000


@dataclass
class ExportConfig:
    """Configuration for data export"""
    export_format: str = 'hdf5'  # 'hdf5', 'pickle', 'npz'
    compression: bool = True
    export_frequency: int = 10000
    max_export_size_mb: int = 1000


@dataclass
class MemoryConfig:
    """Configuration for memory management"""
    enable_memory_efficient_mode: bool = False
    chunk_size: int = 100
    offload_to_cpu: bool = False
    clear_cache_frequency: int = 1000


@dataclass
class AnalysisConfig:
    """
    Complete configuration for Wave Transformer analysis suite.

    This dataclass provides centralized configuration for all analysis components
    including data collection, visualization, introspection, training monitoring,
    and export settings.

    Args:
        output_dir: Base directory for all analysis outputs
        experiment_name: Name for this analysis run
        enable_wave_tracking: Master switch for wave tracking
        wave_sampling_rate: Global sampling rate for wave collection (0.0-1.0)
        collector: Configuration for WaveCollector
        visualization: Configuration for visualization tools
        introspection: Configuration for introspection tools
        training: Configuration for training monitoring
        export: Configuration for data export
        memory: Configuration for memory management
        custom_config: Additional custom configuration

    Example:
        >>> config = AnalysisConfig(
        ...     output_dir='analysis_results',
        ...     experiment_name='wave_experiment_1',
        ...     enable_wave_tracking=True
        ... )
        >>> config.to_yaml('config.yaml')
        >>> loaded_config = AnalysisConfig.from_yaml('config.yaml')
    """

    # General settings
    output_dir: str = 'analysis_results'
    experiment_name: str = 'wave_analysis'
    enable_wave_tracking: bool = True
    wave_sampling_rate: float = 0.1  # Sample 10% of batches by default
    random_seed: int = 42
    device: str = 'cuda'

    # Component configurations
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    introspection: IntrospectionConfig = field(default_factory=IntrospectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Custom configuration
    custom_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate sampling rates
        if not 0.0 <= self.wave_sampling_rate <= 1.0:
            raise ValueError(f"wave_sampling_rate must be in [0.0, 1.0], got {self.wave_sampling_rate}")

        if not 0.0 <= self.collector.sampling_rate <= 1.0:
            raise ValueError(f"collector.sampling_rate must be in [0.0, 1.0], got {self.collector.sampling_rate}")

        # Validate directories
        if not self.output_dir:
            raise ValueError("output_dir cannot be empty")

        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Validate visualization settings
        valid_formats = ['png', 'pdf', 'svg', 'jpg']
        if self.visualization.figure_format not in valid_formats:
            warnings.warn(
                f"Invalid figure_format '{self.visualization.figure_format}'. "
                f"Valid options: {valid_formats}. Using 'png'."
            )
            self.visualization.figure_format = 'png'

        # Validate export format
        valid_export_formats = ['hdf5', 'pickle', 'npz']
        if self.export.export_format not in valid_export_formats:
            warnings.warn(
                f"Invalid export_format '{self.export.export_format}'. "
                f"Valid options: {valid_export_formats}. Using 'hdf5'."
            )
            self.export.export_format = 'hdf5'

        # Check dependencies
        if self.visualization.use_wandb and not YAML_AVAILABLE:
            warnings.warn(
                "wandb logging enabled but wandb not installed. "
                "Install with 'pip install wandb'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return asdict(self)

    def to_yaml(self, filepath: str):
        """
        Save configuration to YAML file.

        Args:
            filepath: Path to output YAML file

        Raises:
            ImportError: If pyyaml is not installed
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "pyyaml is required for YAML support. "
                "Install with 'pip install pyyaml'"
            )

        config_dict = self.to_dict()

        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        print(f"Configuration saved to: {filepath}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AnalysisConfig':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            AnalysisConfig instance
        """
        # Extract nested configs
        collector_dict = config_dict.pop('collector', {})
        visualization_dict = config_dict.pop('visualization', {})
        introspection_dict = config_dict.pop('introspection', {})
        training_dict = config_dict.pop('training', {})
        export_dict = config_dict.pop('export', {})
        memory_dict = config_dict.pop('memory', {})

        # Create nested config objects
        collector = CollectorConfig(**collector_dict)
        visualization = VisualizationConfig(**visualization_dict)
        introspection = IntrospectionConfig(**introspection_dict)
        training = TrainingConfig(**training_dict)
        export = ExportConfig(**export_dict)
        memory = MemoryConfig(**memory_dict)

        # Create main config
        return cls(
            collector=collector,
            visualization=visualization,
            introspection=introspection,
            training=training,
            export=export,
            memory=memory,
            **config_dict
        )

    @classmethod
    def from_yaml(cls, filepath: str) -> 'AnalysisConfig':
        """
        Load configuration from YAML file.

        Args:
            filepath: Path to YAML configuration file

        Returns:
            AnalysisConfig instance

        Raises:
            ImportError: If pyyaml is not installed
            FileNotFoundError: If file doesn't exist
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "pyyaml is required for YAML support. "
                "Install with 'pip install pyyaml'"
            )

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    def update(self, **kwargs):
        """
        Update configuration values.

        Args:
            **kwargs: Key-value pairs to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Unknown configuration key: {key}")

    def get_component_config(self, component: str) -> Any:
        """
        Get configuration for a specific component.

        Args:
            component: Component name ('collector', 'visualization', etc.)

        Returns:
            Component configuration object

        Raises:
            ValueError: If component name is invalid
        """
        valid_components = [
            'collector', 'visualization', 'introspection',
            'training', 'export', 'memory'
        ]

        if component not in valid_components:
            raise ValueError(
                f"Invalid component '{component}'. "
                f"Valid options: {valid_components}"
            )

        return getattr(self, component)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AnalysisConfig(\n"
            f"  experiment_name='{self.experiment_name}',\n"
            f"  output_dir='{self.output_dir}',\n"
            f"  enable_wave_tracking={self.enable_wave_tracking},\n"
            f"  wave_sampling_rate={self.wave_sampling_rate},\n"
            f"  visualization.use_tensorboard={self.visualization.use_tensorboard},\n"
            f"  visualization.use_wandb={self.visualization.use_wandb}\n"
            f")"
        )


def create_default_config(
    output_dir: str = 'analysis_results',
    experiment_name: str = 'wave_analysis',
    enable_tensorboard: bool = True,
    enable_wandb: bool = False
) -> AnalysisConfig:
    """
    Create a default configuration with common settings.

    Args:
        output_dir: Output directory
        experiment_name: Experiment name
        enable_tensorboard: Enable TensorBoard logging
        enable_wandb: Enable Weights & Biases logging

    Returns:
        AnalysisConfig with default settings
    """
    config = AnalysisConfig(
        output_dir=output_dir,
        experiment_name=experiment_name
    )

    config.visualization.use_tensorboard = enable_tensorboard
    config.visualization.use_wandb = enable_wandb

    return config


def create_minimal_config(output_dir: str = 'analysis_results') -> AnalysisConfig:
    """
    Create minimal configuration for lightweight analysis.

    Disables most features for minimal overhead.

    Args:
        output_dir: Output directory

    Returns:
        Minimal AnalysisConfig
    """
    config = AnalysisConfig(
        output_dir=output_dir,
        enable_wave_tracking=True,
        wave_sampling_rate=0.01  # Sample only 1%
    )

    # Disable heavy features
    config.collector.track_gradients = False
    config.collector.store_full_waves = False
    config.visualization.use_tensorboard = False
    config.visualization.use_wandb = False
    config.introspection.enable_interference_analysis = False
    config.introspection.enable_spectrum_tracking = False
    config.training.enable_gradient_monitoring = False
    config.memory.enable_memory_efficient_mode = True

    return config


def create_full_config(
    output_dir: str = 'analysis_results',
    experiment_name: str = 'full_analysis'
) -> AnalysisConfig:
    """
    Create configuration with all features enabled.

    Useful for comprehensive analysis but has higher overhead.

    Args:
        output_dir: Output directory
        experiment_name: Experiment name

    Returns:
        Full-featured AnalysisConfig
    """
    config = AnalysisConfig(
        output_dir=output_dir,
        experiment_name=experiment_name,
        enable_wave_tracking=True,
        wave_sampling_rate=0.5  # Sample 50%
    )

    # Enable all features
    config.collector.track_gradients = True
    config.collector.store_full_waves = True
    config.collector.compute_statistics = True
    config.visualization.use_tensorboard = True
    config.visualization.use_wandb = True
    config.visualization.save_figures = True
    config.introspection.enable_spectrum_tracking = True
    config.introspection.enable_interference_analysis = True
    config.introspection.enable_layer_analysis = True
    config.training.enable_gradient_monitoring = True
    config.training.enable_callbacks = True

    return config
