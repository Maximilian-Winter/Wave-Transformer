"""
Visualization and Reporting Tools for Wave Transformer Analysis

This module provides publication-quality visualization and logging tools for
analyzing Wave Transformer models.

Components:
-----------
- WaveTensorBoardWriter: TensorBoard integration for real-time monitoring
- WaveWandbLogger: Weights & Biases integration for experiment tracking
- PaperReportGenerator: Publication-quality figure and table generation

Example Usage:
--------------
TensorBoard Logging:
    >>> from wave_transformer.analysis.visualization import WaveTensorBoardWriter
    >>> writer = WaveTensorBoardWriter(log_dir='runs/experiment_1')
    >>> writer.add_wave_statistics(wave, tag='encoder/wave', step=100)
    >>> writer.add_wave_heatmaps(wave, tag='encoder/wave', step=100)
    >>> writer.close()

Weights & Biases Logging:
    >>> from wave_transformer.analysis.visualization import WaveWandbLogger
    >>> logger = WaveWandbLogger(project='wave-transformer', name='exp-1')
    >>> logger.log_wave_statistics(wave, prefix='encoder', step=100)
    >>> logger.log_wave_visualizations(wave, prefix='encoder', step=100)
    >>> logger.finish()

Publication Reports:
    >>> from wave_transformer.analysis.visualization import (
    ...     PaperReportGenerator, PublicationStyle
    ... )
    >>> generator = PaperReportGenerator(
    ...     output_dir='paper_figures',
    ...     style=PublicationStyle.IEEE
    ... )
    >>> generator.create_training_curve_figure(
    ...     train_losses=losses,
    ...     save_name='training'
    ... )
    >>> generator.generate_full_report(analysis_results)
"""

from wave_transformer.analysis.visualization.tensorboard_writer import (
    WaveTensorBoardWriter
)
from wave_transformer.analysis.visualization.wandb_logger import (
    WaveWandbLogger
)
from wave_transformer.analysis.visualization.report_generator import (
    PaperReportGenerator,
    PublicationStyle,
    FigureConfig
)

__all__ = [
    'WaveTensorBoardWriter',
    'WaveWandbLogger',
    'PaperReportGenerator',
    'PublicationStyle',
    'FigureConfig',
]
