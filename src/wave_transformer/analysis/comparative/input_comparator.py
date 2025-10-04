"""
Input Comparator

Compares how different inputs are represented in wave space,
enabling clustering, similarity search, and input analysis.
"""

from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

try:
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available. Hierarchical clustering will be limited.")

from wave_transformer.core.wave import Wave


class InputComparator:
    """
    Compare wave representations of different inputs.

    This class provides methods to:
    - Generate wave representations for multiple inputs
    - Compute pairwise similarity matrices
    - Cluster inputs by wave similarity (k-means, hierarchical)
    - Visualize input relationships (heatmaps, t-SNE/UMAP)
    - Find nearest neighbors in wave space

    Args:
        model: WaveTransformer model
        device: Device to perform computations on

    Example:
        >>> comparator = InputComparator(model, device='cuda')
        >>> waves = comparator.compare_inputs(inputs)
        >>> similarity = comparator.compute_input_similarity(waves)
        >>> clusters = comparator.cluster_inputs(waves, method='kmeans', n_clusters=5)
        >>> comparator.plot_input_comparison(similarity)
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def compare_inputs(
        self,
        encoder_inputs: List[Dict[str, Any]],
        attention_masks: Optional[List[torch.Tensor]] = None,
        extract_layer: Optional[str] = 'encoder'
    ) -> List[Wave]:
        """
        Generate wave representations for multiple inputs.

        Args:
            encoder_inputs: List of input dictionaries for encoder
            attention_masks: Optional list of attention masks
            extract_layer: Which layer to extract waves from:
                          'encoder' (default), 'layer_N', or 'pre_decoder'

        Returns:
            List of Wave objects, one per input
        """
        self.model.eval()
        waves = []

        if attention_masks is None:
            attention_masks = [None] * len(encoder_inputs)

        for encoder_input, attention_mask in zip(encoder_inputs, attention_masks):
            # Move input to device
            encoder_input_device = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in encoder_input.items()
            }

            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            if extract_layer == 'encoder':
                # Extract from encoder
                wave = self.model.wave_encoder(
                    attention_mask=attention_mask,
                    **encoder_input_device
                )
            else:
                # Extract from transformer layers using hooks
                layer_outputs = {}

                def make_hook(name):
                    def hook(module, input, output):
                        layer_outputs[name] = output
                    return hook

                # Register hook
                if extract_layer.startswith('layer_'):
                    layer_idx = int(extract_layer.split('_')[1])
                    handle = self.model.layers[layer_idx].register_forward_hook(
                        make_hook(extract_layer)
                    )
                elif extract_layer == 'pre_decoder':
                    handle = self.model.norm_f.register_forward_hook(
                        make_hook('pre_decoder')
                    )
                else:
                    raise ValueError(f"Unknown layer: {extract_layer}")

                # Forward pass
                _ = self.model(encoder_input_device, attention_mask=attention_mask)

                # Extract wave from representation
                wave = Wave.from_representation(layer_outputs[extract_layer])

                # Remove hook
                handle.remove()

            waves.append(wave)

        return waves

    def compute_input_similarity(
        self,
        waves: List[Wave],
        method: str = 'cosine',
        batch_idx: int = 0
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix between inputs.

        Args:
            waves: List of Wave objects from compare_inputs()
            method: Similarity metric:
                   - 'cosine': Cosine similarity of flattened wave tensors
                   - 'correlation': Correlation of harmonic patterns
                   - 'spectral_overlap': Frequency domain overlap
                   - 'l2': Negative L2 distance (higher = more similar)
            batch_idx: Which batch element to use (default: 0)

        Returns:
            Similarity matrix of shape [n_inputs, n_inputs]
            Values typically in [-1, 1] for cosine/correlation, or [0, inf) for others
        """
        n = len(waves)
        similarity_matrix = np.zeros((n, n))

        # Extract representations
        representations = []
        for wave in waves:
            repr = wave.to_representation()[batch_idx].detach().cpu().numpy().flatten()
            representations.append(repr)

        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                    continue

                repr_i = representations[i]
                repr_j = representations[j]

                if method == 'cosine':
                    # Cosine similarity
                    dot_product = np.dot(repr_i, repr_j)
                    norm_i = np.linalg.norm(repr_i)
                    norm_j = np.linalg.norm(repr_j)
                    similarity = dot_product / (norm_i * norm_j + 1e-8)

                elif method == 'correlation':
                    # Pearson correlation
                    similarity = np.corrcoef(repr_i, repr_j)[0, 1]

                elif method == 'spectral_overlap':
                    # Frequency domain overlap using amplitude-weighted frequencies
                    wave_i = waves[i]
                    wave_j = waves[j]

                    freq_i = wave_i.frequencies[batch_idx].detach().cpu().numpy().flatten()
                    freq_j = wave_j.frequencies[batch_idx].detach().cpu().numpy().flatten()
                    amp_i = wave_i.amplitudes[batch_idx].detach().cpu().numpy().flatten()
                    amp_j = wave_j.amplitudes[batch_idx].detach().cpu().numpy().flatten()

                    # Normalize amplitudes
                    amp_i = amp_i / (amp_i.sum() + 1e-8)
                    amp_j = amp_j / (amp_j.sum() + 1e-8)

                    # Compute weighted frequency overlap
                    freq_diff = np.abs(freq_i - freq_j)
                    overlap = np.exp(-freq_diff).mean()  # Gaussian-like overlap
                    similarity = overlap

                elif method == 'l2':
                    # Negative L2 distance (higher = more similar)
                    l2_dist = np.linalg.norm(repr_i - repr_j)
                    similarity = -l2_dist

                else:
                    raise ValueError(f"Unknown similarity method: {method}")

                similarity_matrix[i, j] = similarity

        return similarity_matrix

    def cluster_inputs(
        self,
        waves: List[Wave],
        method: str = 'kmeans',
        n_clusters: int = 5,
        batch_idx: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Cluster inputs by wave representation similarity.

        Args:
            waves: List of Wave objects from compare_inputs()
            method: Clustering method:
                   - 'kmeans': K-means clustering
                   - 'hierarchical': Hierarchical agglomerative clustering
            n_clusters: Number of clusters
            batch_idx: Which batch element to use
            **kwargs: Additional arguments for clustering algorithm

        Returns:
            Dictionary containing:
            - 'labels': Cluster labels for each input
            - 'method': Clustering method used
            - 'n_clusters': Number of clusters
            - 'linkage_matrix': (for hierarchical only) linkage matrix
        """
        # Extract flattened representations
        representations = []
        for wave in waves:
            repr = wave.to_representation()[batch_idx].detach().cpu().numpy().flatten()
            representations.append(repr)

        X = np.array(representations)

        if method == 'kmeans':
            try:
                from sklearn.cluster import KMeans
            except ImportError:
                raise ImportError("sklearn required for k-means clustering. "
                                "Install with: pip install scikit-learn")

            kmeans = KMeans(n_clusters=n_clusters, **kwargs)
            labels = kmeans.fit_predict(X)

            return {
                'labels': labels,
                'method': 'kmeans',
                'n_clusters': n_clusters,
                'cluster_centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_
            }

        elif method == 'hierarchical':
            if not HAS_SCIPY:
                raise ImportError("scipy required for hierarchical clustering. "
                                "Install with: pip install scipy")

            # Compute linkage matrix
            linkage_method = kwargs.get('linkage', 'ward')
            Z = linkage(X, method=linkage_method)

            # Get cluster labels
            labels = fcluster(Z, n_clusters, criterion='maxclust')
            labels = labels - 1  # Convert to 0-indexed

            return {
                'labels': labels,
                'method': 'hierarchical',
                'n_clusters': n_clusters,
                'linkage_matrix': Z,
                'linkage_method': linkage_method
            }

        else:
            raise ValueError(f"Unknown clustering method: {method}")

    def plot_input_comparison(
        self,
        similarity_matrix: np.ndarray,
        input_labels: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot similarity matrix as a heatmap.

        Args:
            similarity_matrix: Output from compute_input_similarity()
            input_labels: Optional labels for inputs
            figsize: Figure size
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        n_inputs = similarity_matrix.shape[0]

        if input_labels is None:
            input_labels = [f'Input {i}' for i in range(n_inputs)]

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            similarity_matrix,
            annot=True if n_inputs <= 20 else False,
            fmt='.2f',
            cmap='RdYlBu',
            xticklabels=input_labels,
            yticklabels=input_labels,
            ax=ax,
            cbar_kws={'label': 'Similarity'},
            vmin=-1 if similarity_matrix.min() < 0 else 0,
            vmax=1
        )

        ax.set_title('Input Similarity Matrix (Wave Space)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_clustering_results(
        self,
        cluster_result: Dict[str, Any],
        waves: List[Wave],
        input_labels: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize clustering results.

        For hierarchical clustering: shows dendrogram
        For k-means: shows cluster assignments and optionally 2D projection

        Args:
            cluster_result: Output from cluster_inputs()
            waves: Original wave objects
            input_labels: Optional labels for inputs
            figsize: Figure size
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        labels = cluster_result['labels']
        n_inputs = len(labels)

        if input_labels is None:
            input_labels = [f'Input {i}' for i in range(n_inputs)]

        if cluster_result['method'] == 'hierarchical':
            # Plot dendrogram
            fig, ax = plt.subplots(figsize=figsize)

            dendrogram(
                cluster_result['linkage_matrix'],
                labels=input_labels,
                ax=ax,
                color_threshold=0
            )

            ax.set_title(f'Hierarchical Clustering Dendrogram '
                        f'(method={cluster_result["linkage_method"]})',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Input')
            ax.set_ylabel('Distance')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

        else:  # kmeans
            # Create figure with cluster assignment visualization
            fig = plt.figure(figsize=figsize)

            # Try to create 2D projection if sklearn is available
            try:
                from sklearn.manifold import TSNE

                # Extract representations for dimensionality reduction
                representations = []
                for wave in waves:
                    repr = wave.to_representation()[0].detach().cpu().numpy().flatten()
                    representations.append(repr)
                X = np.array(representations)

                # Reduce to 2D with t-SNE
                tsne = TSNE(n_components=2, random_state=42)
                X_2d = tsne.fit_transform(X)

                # Plot
                ax = fig.add_subplot(111)
                scatter = ax.scatter(
                    X_2d[:, 0], X_2d[:, 1],
                    c=labels, cmap='tab10',
                    s=100, alpha=0.7, edgecolors='black'
                )

                # Add labels
                for i, label in enumerate(input_labels):
                    ax.annotate(label, (X_2d[i, 0], X_2d[i, 1]),
                              fontsize=8, alpha=0.7)

                ax.set_title(f'K-means Clustering (k={cluster_result["n_clusters"]}) '
                           f'- t-SNE Projection',
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('t-SNE Dimension 1')
                ax.set_ylabel('t-SNE Dimension 2')
                ax.grid(True, alpha=0.3)

                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Cluster')

            except ImportError:
                # Fallback: just show cluster assignments as bar chart
                ax = fig.add_subplot(111)
                colors = plt.cm.tab10(labels / max(labels))
                ax.barh(range(n_inputs), labels, color=colors)
                ax.set_yticks(range(n_inputs))
                ax.set_yticklabels(input_labels)
                ax.set_xlabel('Cluster')
                ax.set_title(f'K-means Cluster Assignments (k={cluster_result["n_clusters"]})',
                           fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_2d_projection(
        self,
        waves: List[Wave],
        method: str = 'tsne',
        labels: Optional[np.ndarray] = None,
        input_labels: Optional[List[str]] = None,
        batch_idx: int = 0,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Create 2D projection of wave representations using t-SNE or UMAP.

        Args:
            waves: List of Wave objects
            method: Dimensionality reduction method ('tsne' or 'umap')
            labels: Optional cluster labels for coloring
            input_labels: Optional text labels for inputs
            batch_idx: Which batch element to use
            figsize: Figure size
            save_path: Optional path to save figure
            **kwargs: Additional arguments for t-SNE/UMAP

        Returns:
            Figure object
        """
        # Extract representations
        representations = []
        for wave in waves:
            repr = wave.to_representation()[batch_idx].detach().cpu().numpy().flatten()
            representations.append(repr)
        X = np.array(representations)

        # Perform dimensionality reduction
        if method == 'tsne':
            try:
                from sklearn.manifold import TSNE
            except ImportError:
                raise ImportError("sklearn required for t-SNE. "
                                "Install with: pip install scikit-learn")

            reducer = TSNE(n_components=2, **kwargs)
            X_2d = reducer.fit_transform(X)

        elif method == 'umap':
            try:
                import umap
            except ImportError:
                raise ImportError("umap-learn required for UMAP. "
                                "Install with: pip install umap-learn")

            reducer = umap.UMAP(n_components=2, **kwargs)
            X_2d = reducer.fit_transform(X)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'umap'")

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        if labels is not None:
            scatter = ax.scatter(
                X_2d[:, 0], X_2d[:, 1],
                c=labels, cmap='tab10',
                s=100, alpha=0.7, edgecolors='black'
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Cluster')
        else:
            ax.scatter(
                X_2d[:, 0], X_2d[:, 1],
                s=100, alpha=0.7, edgecolors='black'
            )

        # Add input labels if provided
        if input_labels is not None:
            for i, label in enumerate(input_labels):
                ax.annotate(label, (X_2d[i, 0], X_2d[i, 1]),
                          fontsize=8, alpha=0.7)

        ax.set_title(f'{method.upper()} Projection of Wave Representations',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} Dimension 1')
        ax.set_ylabel(f'{method.upper()} Dimension 2')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def find_nearest_neighbors(
        self,
        waves: List[Wave],
        query_idx: int,
        k: int = 5,
        metric: str = 'cosine',
        batch_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Find k nearest neighbors to a query input in wave space.

        Args:
            waves: List of Wave objects
            query_idx: Index of query input
            k: Number of nearest neighbors to find
            metric: Distance metric ('cosine', 'l2', 'correlation')
            batch_idx: Which batch element to use

        Returns:
            Dictionary containing:
            - 'neighbor_indices': Indices of k nearest neighbors
            - 'distances': Distances to neighbors
            - 'query_idx': Original query index
        """
        # Compute similarity matrix
        similarity_matrix = self.compute_input_similarity(
            waves, method=metric, batch_idx=batch_idx
        )

        # For cosine/correlation, convert similarity to distance
        if metric in ['cosine', 'correlation']:
            distance_matrix = 1 - similarity_matrix
        else:
            distance_matrix = -similarity_matrix  # For l2, we stored negative distance

        # Get distances from query
        distances = distance_matrix[query_idx]

        # Sort by distance (excluding self)
        sorted_indices = np.argsort(distances)
        # Remove query itself from results
        sorted_indices = sorted_indices[sorted_indices != query_idx]

        # Get k nearest
        neighbor_indices = sorted_indices[:k]
        neighbor_distances = distances[neighbor_indices]

        return {
            'query_idx': query_idx,
            'neighbor_indices': neighbor_indices.tolist(),
            'distances': neighbor_distances.tolist(),
            'metric': metric
        }

    def compare_wave_statistics(
        self,
        waves: List[Wave],
        input_labels: Optional[List[str]] = None,
        batch_idx: int = 0,
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, Dict[str, List]]:
        """
        Compare statistical properties of waves across inputs.

        Args:
            waves: List of Wave objects
            input_labels: Optional labels for inputs
            batch_idx: Which batch element to use
            figsize: Figure size
            save_path: Optional path to save figure

        Returns:
            Figure and dictionary of statistics
        """
        n_inputs = len(waves)

        if input_labels is None:
            input_labels = [f'Input {i}' for i in range(n_inputs)]

        # Collect statistics
        stats = {
            'mean_amplitude': [],
            'mean_frequency': [],
            'total_energy': [],
            'spectral_centroid': [],
            'phase_std': []
        }

        for wave in waves:
            freqs = wave.frequencies[batch_idx].detach().cpu().numpy()
            amps = wave.amplitudes[batch_idx].detach().cpu().numpy()
            phases = wave.phases[batch_idx].detach().cpu().numpy()

            stats['mean_amplitude'].append(amps.mean())
            stats['mean_frequency'].append(freqs.mean())
            stats['total_energy'].append((amps ** 2).sum())
            stats['phase_std'].append(phases.std())

            # Spectral centroid: amplitude-weighted mean frequency
            total_amp = amps.sum() + 1e-8
            spectral_centroid = (freqs * amps).sum() / total_amp
            stats['spectral_centroid'].append(spectral_centroid)

        # Plot
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        for idx, (stat_name, values) in enumerate(stats.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]
            ax.bar(range(n_inputs), values, alpha=0.7)
            ax.set_title(stat_name.replace('_', ' ').title(), fontweight='bold')
            ax.set_xlabel('Input')
            ax.set_ylabel(stat_name.replace('_', ' ').title())
            ax.set_xticks(range(n_inputs))
            ax.set_xticklabels(input_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')

        # Hide unused subplot
        if len(stats) < len(axes):
            axes[-1].axis('off')

        fig.suptitle('Wave Statistics Comparison Across Inputs',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, stats
