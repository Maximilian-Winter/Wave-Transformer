"""
Export utilities for analysis results.

This module provides comprehensive export functionality for wave analysis results,
supporting multiple formats (JSON, HDF5, TensorBoard, W&B) and publication-quality
figure generation.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy and torch types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


class AnalysisExporter:
    """
    Comprehensive exporter for wave analysis results.

    Supports multiple export formats and automatic format detection.
    All methods are static for ease of use.
    """

    @staticmethod
    def to_json(
        data: Dict[str, Any],
        filepath: Union[str, Path],
        indent: int = 2,
        compress: bool = False
    ) -> None:
        """
        Export data to JSON format with numpy/torch handling.

        Args:
            data: Dictionary of data to export
            filepath: Output file path
            indent: JSON indentation level
            compress: If True, use gzip compression
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert data to JSON-serializable format
        json_data = json.dumps(data, cls=NumpyEncoder, indent=indent)

        if compress:
            import gzip
            with gzip.open(f"{filepath}.gz", 'wt', encoding='utf-8') as f:
                f.write(json_data)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_data)

    @staticmethod
    def from_json(
        filepath: Union[str, Path],
        decompress: bool = False
    ) -> Dict[str, Any]:
        """
        Load data from JSON file.

        Args:
            filepath: Input file path
            decompress: If True, decompress gzip file

        Returns:
            Dictionary of loaded data
        """
        filepath = Path(filepath)

        if decompress or str(filepath).endswith('.gz'):
            import gzip
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

        return data

    @staticmethod
    def to_hdf5(
        data: Dict[str, Any],
        filepath: Union[str, Path],
        compression: str = 'gzip',
        compression_opts: int = 4
    ) -> None:
        """
        Export large datasets to HDF5 format.

        Args:
            data: Dictionary of data to export
            filepath: Output file path
            compression: Compression algorithm ('gzip', 'lzf', None)
            compression_opts: Compression level (0-9 for gzip)
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 export. Install with: pip install h5py"
            )

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, 'w') as f:
            AnalysisExporter._write_dict_to_hdf5(
                f, data, compression=compression, compression_opts=compression_opts
            )

    @staticmethod
    def _write_dict_to_hdf5(
        group: 'h5py.Group',
        data: Dict[str, Any],
        compression: Optional[str] = 'gzip',
        compression_opts: int = 4
    ) -> None:
        """
        Recursively write dictionary to HDF5 group.

        Args:
            group: HDF5 group to write to
            data: Dictionary data
            compression: Compression algorithm
            compression_opts: Compression level
        """
        for key, value in data.items():
            # Sanitize key for HDF5
            safe_key = str(key).replace('/', '_')

            if isinstance(value, dict):
                # Create subgroup for nested dict
                subgroup = group.create_group(safe_key)
                AnalysisExporter._write_dict_to_hdf5(
                    subgroup, value, compression, compression_opts
                )
            elif isinstance(value, (list, tuple)):
                # Try to convert to array
                try:
                    arr = np.array(value)
                    group.create_dataset(
                        safe_key, data=arr,
                        compression=compression,
                        compression_opts=compression_opts
                    )
                except (ValueError, TypeError):
                    # Fall back to string representation
                    group.attrs[safe_key] = str(value)
            elif isinstance(value, np.ndarray):
                group.create_dataset(
                    safe_key, data=value,
                    compression=compression,
                    compression_opts=compression_opts
                )
            elif isinstance(value, torch.Tensor):
                arr = value.detach().cpu().numpy()
                group.create_dataset(
                    safe_key, data=arr,
                    compression=compression,
                    compression_opts=compression_opts
                )
            elif isinstance(value, (int, float, str, bool)):
                group.attrs[safe_key] = value
            elif value is None:
                group.attrs[safe_key] = 'None'
            else:
                # Store as string representation
                group.attrs[safe_key] = str(value)

    @staticmethod
    def from_hdf5(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load data from HDF5 file.

        Args:
            filepath: Input file path

        Returns:
            Dictionary of loaded data
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 import. Install with: pip install h5py"
            )

        filepath = Path(filepath)

        with h5py.File(filepath, 'r') as f:
            return AnalysisExporter._read_hdf5_to_dict(f)

    @staticmethod
    def _read_hdf5_to_dict(group: 'h5py.Group') -> Dict[str, Any]:
        """
        Recursively read HDF5 group to dictionary.

        Args:
            group: HDF5 group to read from

        Returns:
            Dictionary of data
        """
        data = {}

        # Read attributes
        for key, value in group.attrs.items():
            data[key] = value

        # Read datasets and subgroups
        for key in group.keys():
            item = group[key]
            if isinstance(item, type(group)):  # It's a subgroup
                data[key] = AnalysisExporter._read_hdf5_to_dict(item)
            else:  # It's a dataset
                data[key] = item[()]

        return data

    @staticmethod
    def to_tensorboard(
        data: Dict[str, Any],
        writer: 'SummaryWriter',
        step: Optional[int] = None,
        prefix: str = ''
    ) -> None:
        """
        Write data to TensorBoard.

        Args:
            data: Dictionary of data to log
            writer: TensorBoard SummaryWriter instance
            step: Global step value
            prefix: Prefix for all tags
        """
        for key, value in data.items():
            tag = f"{prefix}/{key}" if prefix else key

            if isinstance(value, dict):
                # Recursively log nested dicts
                AnalysisExporter.to_tensorboard(
                    value, writer, step=step, prefix=tag
                )
            elif isinstance(value, (int, float, np.integer, np.floating)):
                # Log scalar
                writer.add_scalar(tag, float(value), step)
            elif isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    # Scalar tensor
                    writer.add_scalar(tag, value.item(), step)
                elif value.ndim == 1:
                    # Histogram
                    writer.add_histogram(tag, value, step)
                elif value.ndim == 2:
                    # Image or heatmap
                    writer.add_image(tag, value.unsqueeze(0), step)
            elif isinstance(value, np.ndarray):
                if value.ndim == 0:
                    writer.add_scalar(tag, float(value), step)
                elif value.ndim == 1:
                    writer.add_histogram(tag, value, step)
                elif value.ndim == 2:
                    # Convert to image format
                    writer.add_image(tag, value[np.newaxis, :, :], step)
            elif isinstance(value, (list, tuple)):
                # Try to convert to array
                try:
                    arr = np.array(value)
                    if arr.dtype in [np.float32, np.float64, np.int32, np.int64]:
                        writer.add_histogram(tag, arr, step)
                except (ValueError, TypeError):
                    pass

    @staticmethod
    def to_wandb(
        data: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = '',
        commit: bool = True
    ) -> None:
        """
        Log data to Weights & Biases.

        Args:
            data: Dictionary of data to log
            step: Global step value
            prefix: Prefix for all keys
            commit: Whether to commit the log immediately
        """
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is required for W&B logging. Install with: pip install wandb"
            )

        if not wandb.run:
            warnings.warn("No active W&B run. Call wandb.init() first.")
            return

        # Flatten nested dictionaries
        flat_data = AnalysisExporter._flatten_dict(data, prefix=prefix)

        # Convert to wandb-compatible format
        wandb_data = {}
        for key, value in flat_data.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                wandb_data[key] = float(value)
            elif isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    wandb_data[key] = value.item()
                elif value.ndim <= 3:
                    # Log as histogram or image
                    wandb_data[key] = wandb.Histogram(value.detach().cpu().numpy())
            elif isinstance(value, np.ndarray):
                if value.ndim == 0:
                    wandb_data[key] = float(value)
                elif value.ndim == 1:
                    wandb_data[key] = wandb.Histogram(value)
                elif value.ndim == 2:
                    wandb_data[key] = wandb.Image(value)
            elif isinstance(value, (list, tuple)):
                try:
                    arr = np.array(value)
                    if arr.dtype in [np.float32, np.float64, np.int32, np.int64]:
                        wandb_data[key] = wandb.Histogram(arr)
                except (ValueError, TypeError):
                    pass

        wandb.log(wandb_data, step=step, commit=commit)

    @staticmethod
    def _flatten_dict(
        data: Dict[str, Any],
        prefix: str = '',
        separator: str = '/'
    ) -> Dict[str, Any]:
        """
        Flatten nested dictionary with prefix.

        Args:
            data: Dictionary to flatten
            prefix: Prefix for keys
            separator: Separator between levels

        Returns:
            Flattened dictionary
        """
        flat = {}

        for key, value in data.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key

            if isinstance(value, dict):
                flat.update(
                    AnalysisExporter._flatten_dict(value, prefix=new_key, separator=separator)
                )
            else:
                flat[new_key] = value

        return flat

    @staticmethod
    def create_paper_figure(
        figsize: Tuple[int, int] = (8, 6),
        dpi: int = 300,
        style: str = 'seaborn-v0_8-paper',
        font_family: str = 'serif',
        font_size: int = 10,
        use_latex: bool = False
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create publication-quality figure with proper styling.

        Args:
            figsize: Figure size in inches
            dpi: Dots per inch
            style: Matplotlib style
            font_family: Font family
            font_size: Base font size
            use_latex: Whether to use LaTeX for text rendering

        Returns:
            Tuple of (figure, axes)
        """
        # Set style
        available_styles = plt.style.available
        if style in available_styles:
            plt.style.use(style)
        else:
            warnings.warn(f"Style '{style}' not available. Using default.")

        # Configure matplotlib for publication quality
        mpl.rcParams['figure.dpi'] = dpi
        mpl.rcParams['savefig.dpi'] = dpi
        mpl.rcParams['font.family'] = font_family
        mpl.rcParams['font.size'] = font_size
        mpl.rcParams['axes.labelsize'] = font_size
        mpl.rcParams['axes.titlesize'] = font_size + 2
        mpl.rcParams['xtick.labelsize'] = font_size - 1
        mpl.rcParams['ytick.labelsize'] = font_size - 1
        mpl.rcParams['legend.fontsize'] = font_size - 1
        mpl.rcParams['figure.titlesize'] = font_size + 4

        if use_latex:
            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        return fig, ax

    @staticmethod
    def save_figure(
        fig: plt.Figure,
        filepath: Union[str, Path],
        formats: Optional[List[str]] = None,
        tight_layout: bool = True,
        bbox_inches: str = 'tight',
        pad_inches: float = 0.1,
        metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Save figure in multiple formats with metadata.

        Args:
            fig: Matplotlib figure to save
            filepath: Base filepath (without extension)
            formats: List of formats to save (['png', 'pdf', 'svg'])
            tight_layout: Whether to apply tight layout
            bbox_inches: Bounding box mode
            pad_inches: Padding around figure
            metadata: Metadata to embed in file
        """
        if formats is None:
            formats = ['png']

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Remove extension if present
        base_path = filepath.with_suffix('')

        if tight_layout:
            fig.tight_layout()

        # Add metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            'Creator': 'Wave Transformer Analysis Suite',
            'CreationDate': datetime.now().isoformat()
        })

        for fmt in formats:
            output_path = f"{base_path}.{fmt}"

            if fmt == 'png':
                fig.savefig(
                    output_path,
                    format='png',
                    bbox_inches=bbox_inches,
                    pad_inches=pad_inches,
                    metadata=metadata
                )
            elif fmt == 'pdf':
                fig.savefig(
                    output_path,
                    format='pdf',
                    bbox_inches=bbox_inches,
                    pad_inches=pad_inches,
                    metadata=metadata
                )
            elif fmt == 'svg':
                fig.savefig(
                    output_path,
                    format='svg',
                    bbox_inches=bbox_inches,
                    pad_inches=pad_inches,
                    metadata=metadata
                )
            elif fmt == 'eps':
                fig.savefig(
                    output_path,
                    format='eps',
                    bbox_inches=bbox_inches,
                    pad_inches=pad_inches
                )
            else:
                warnings.warn(f"Unknown format: {fmt}")

    @staticmethod
    def export_report(
        data: Dict[str, Any],
        output_dir: Union[str, Path],
        formats: List[str] = ['json', 'hdf5'],
        prefix: str = 'analysis_report'
    ) -> Dict[str, Path]:
        """
        Export comprehensive analysis report in multiple formats.

        Args:
            data: Analysis data to export
            output_dir: Output directory
            formats: List of formats to export to
            prefix: Filename prefix

        Returns:
            Dictionary mapping format names to output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_files = {}

        for fmt in formats:
            if fmt == 'json':
                filepath = output_dir / f"{prefix}.json"
                AnalysisExporter.to_json(data, filepath)
                output_files['json'] = filepath
            elif fmt == 'hdf5':
                filepath = output_dir / f"{prefix}.h5"
                AnalysisExporter.to_hdf5(data, filepath)
                output_files['hdf5'] = filepath
            else:
                warnings.warn(f"Unknown export format: {fmt}")

        return output_files

    @staticmethod
    def create_comparison_plot(
        data_dict: Dict[str, np.ndarray],
        title: str = "Comparison",
        xlabel: str = "Step",
        ylabel: str = "Value",
        figsize: Tuple[int, int] = (10, 6),
        style: str = 'default',
        save_path: Optional[Union[str, Path]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create comparison plot for multiple metrics.

        Args:
            data_dict: Dictionary mapping labels to data arrays
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            style: Plot style
            save_path: Optional path to save figure

        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = AnalysisExporter.create_paper_figure(
            figsize=figsize,
            style=style
        )

        for label, data in data_dict.items():
            steps = np.arange(len(data))
            ax.plot(steps, data, label=label, marker='o', markersize=3, alpha=0.8)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path is not None:
            AnalysisExporter.save_figure(fig, save_path)

        return fig, ax
