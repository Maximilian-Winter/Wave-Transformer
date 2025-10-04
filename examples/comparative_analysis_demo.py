"""
Comparative Analysis Tools - Demonstration Script

This script demonstrates the usage of the three comparative analysis tools:
1. CheckpointComparator - Compare training checkpoints
2. InputComparator - Compare input representations
3. AblationHelper - Systematic ablation studies

Note: This is a demonstration script. Adjust paths and parameters for your use case.
"""

import torch
import numpy as np
from pathlib import Path

# Example usage - adjust these imports based on your setup
from wave_transformer.analysis import (
    CheckpointComparator,
    InputComparator,
    AblationHelper
)

# Uncomment these if you have the actual model classes
# from wave_transformer.language_modelling import TokenToWaveEncoder, WaveToTokenDecoder
# from wave_transformer.core.transformer import WaveTransformer


def demo_checkpoint_comparison():
    """
    Demonstrate checkpoint comparison functionality.

    This compares multiple model checkpoints to analyze training evolution.
    """
    print("\n" + "=" * 70)
    print("DEMO 1: Checkpoint Comparison")
    print("=" * 70)

    # Define checkpoint paths (adjust to your actual checkpoint locations)
    checkpoint_paths = [
        'checkpoints/step_1000',
        'checkpoints/step_2000',
        'checkpoints/step_3000',
        'checkpoints/step_4000',
    ]

    print(f"\nCheckpoint paths: {checkpoint_paths}")
    print("Note: This demo requires actual checkpoints. Skipping if not found.")

    # Check if checkpoints exist
    if not all(Path(p).exists() for p in checkpoint_paths):
        print("⚠ Checkpoints not found. Skipping this demo.")
        return

    # Load comparator (requires encoder_cls and decoder_cls)
    # comparator = CheckpointComparator(
    #     checkpoint_paths=checkpoint_paths,
    #     encoder_cls=TokenToWaveEncoder,
    #     decoder_cls=WaveToTokenDecoder,
    #     device='cuda' if torch.cuda.is_available() else 'cpu'
    # )
    #
    # # Prepare sample input
    # input_data = {
    #     'token_ids': torch.tensor([[1, 2, 3, 4, 5]])
    # }
    #
    # # Compute divergence between consecutive checkpoints
    # print("\nComputing checkpoint divergences...")
    # divergence = comparator.compute_checkpoint_divergence(
    #     encoder_input=input_data,
    #     metrics=['l2', 'cosine', 'kl']  # Skip 'wasserstein' if scipy not available
    # )
    #
    # print("\nDivergence results:")
    # for metric, values in divergence.items():
    #     print(f"  {metric}: {values}")
    #
    # # Visualize evolution
    # print("\nGenerating evolution plot...")
    # fig, axes = comparator.plot_checkpoint_evolution(
    #     divergence,
    #     save_path='checkpoint_evolution.png'
    # )
    #
    # # Identify critical checkpoints
    # critical = comparator.identify_critical_checkpoints(
    #     divergence,
    #     threshold_percentile=75.0
    # )
    # print(f"\nCritical checkpoints: {critical}")

    print("\n✓ Checkpoint comparison demo complete (would run with actual checkpoints)")


def demo_input_comparison():
    """
    Demonstrate input comparison functionality.

    This compares how different inputs are represented in wave space.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Input Comparison")
    print("=" * 70)

    print("\nNote: This demo requires a trained model. Skipping if not found.")

    # Load model (adjust to your actual model)
    # model = WaveTransformer.load(
    #     'checkpoints/best_model',
    #     encoder_cls=TokenToWaveEncoder,
    #     decoder_cls=WaveToTokenDecoder
    # )
    # model.eval()
    #
    # # Initialize comparator
    # comparator = InputComparator(
    #     model,
    #     device='cuda' if torch.cuda.is_available() else 'cpu'
    # )
    #
    # # Prepare multiple inputs
    # inputs = [
    #     {'token_ids': torch.tensor([[1, 2, 3, 4]])},
    #     {'token_ids': torch.tensor([[5, 6, 7, 8]])},
    #     {'token_ids': torch.tensor([[1, 2, 5, 6]])},
    #     {'token_ids': torch.tensor([[3, 4, 7, 8]])},
    # ]
    #
    # input_labels = ['Input A', 'Input B', 'Input C', 'Input D']
    #
    # # Generate wave representations
    # print("\nGenerating wave representations...")
    # waves = comparator.compare_inputs(inputs, extract_layer='encoder')
    #
    # # Compute similarity matrix
    # print("\nComputing similarity matrix...")
    # similarity = comparator.compute_input_similarity(
    #     waves,
    #     method='cosine'
    # )
    #
    # print("Similarity matrix:")
    # print(similarity)
    #
    # # Visualize similarity
    # print("\nGenerating similarity heatmap...")
    # fig = comparator.plot_input_comparison(
    #     similarity,
    #     input_labels=input_labels,
    #     save_path='input_similarity.png'
    # )
    #
    # # Cluster inputs
    # print("\nClustering inputs...")
    # cluster_result = comparator.cluster_inputs(
    #     waves,
    #     method='kmeans',
    #     n_clusters=2
    # )
    # print(f"Cluster labels: {cluster_result['labels']}")
    #
    # # Visualize clustering
    # fig = comparator.plot_clustering_results(
    #     cluster_result,
    #     waves,
    #     input_labels=input_labels,
    #     save_path='input_clusters.png'
    # )
    #
    # # Find nearest neighbors
    # neighbors = comparator.find_nearest_neighbors(
    #     waves,
    #     query_idx=0,
    #     k=2,
    #     metric='cosine'
    # )
    # print(f"\nNearest neighbors to Input A: {neighbors['neighbor_indices']}")
    # print(f"Distances: {neighbors['distances']}")
    #
    # # Compare statistics
    # fig, stats = comparator.compare_wave_statistics(
    #     waves,
    #     input_labels=input_labels,
    #     save_path='input_statistics.png'
    # )

    print("\n✓ Input comparison demo complete (would run with actual model)")


def demo_ablation_study():
    """
    Demonstrate ablation study functionality.

    This shows how to systematically ablate model components.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Ablation Study")
    print("=" * 70)

    print("\nNote: This demo requires a trained model and dataloader. Skipping if not found.")

    # Load model
    # model = WaveTransformer.load(
    #     'checkpoints/best_model',
    #     encoder_cls=TokenToWaveEncoder,
    #     decoder_cls=WaveToTokenDecoder
    # )
    #
    # # Initialize ablation helper
    # ablator = AblationHelper(
    #     model,
    #     device='cuda' if torch.cuda.is_available() else 'cpu'
    # )
    #
    # # Define ablation configurations
    # ablation_configs = [
    #     # Ablate first 3 harmonics
    #     {
    #         'type': 'harmonics',
    #         'indices': [0, 1, 2],
    #         'mode': 'zero',
    #         'component': 'all'
    #     },
    #
    #     # Ablate middle harmonics
    #     {
    #         'type': 'harmonics',
    #         'indices': list(range(16, 32)),
    #         'mode': 'random',
    #         'component': 'amp'
    #     },
    #
    #     # Make layer 2 an identity mapping
    #     {
    #         'type': 'layer',
    #         'layer_idx': 2,
    #         'mode': 'identity'
    #     },
    #
    #     # Zero out all phase information
    #     {
    #         'type': 'wave_component',
    #         'component': 'phase',
    #         'mode': 'zero'
    #     },
    #
    #     # Randomize all frequencies
    #     {
    #         'type': 'wave_component',
    #         'component': 'freq',
    #         'mode': 'random'
    #     },
    # ]
    #
    # print(f"\nRunning {len(ablation_configs)} ablation configurations...")
    #
    # # Run ablation study
    # # Note: You need to provide a dataloader or custom eval function
    # # results = ablator.run_ablation_study(
    # #     ablation_configs=ablation_configs,
    # #     dataloader=val_loader,
    # #     include_baseline=True
    # # )
    # #
    # # print("\nAblation results:")
    # # print(results)
    # #
    # # # Visualize results
    # # print("\nGenerating ablation results plot...")
    # # fig = ablator.plot_ablation_results(
    # #     results,
    # #     save_path='ablation_results.png'
    # # )
    # #
    # # # Impact heatmap
    # # fig = ablator.plot_impact_heatmap(
    # #     results,
    # #     metric='avg_loss',
    # #     save_path='ablation_heatmap.png'
    # # )

    # Example of using context manager
    print("\nExample: Using ablation helper with context manager...")
    print("""
    with AblationHelper(model) as ablator:
        # Ablate specific harmonics
        ablator.ablate_harmonics([0, 1, 2], mode='zero')

        # Evaluate ablated model
        # loss = evaluate(model, dataloader)
        # print(f"Loss with ablation: {loss}")

    # Model automatically restored here
    """)

    print("\n✓ Ablation study demo complete (would run with actual model and data)")


def demonstrate_advanced_usage():
    """
    Show advanced usage patterns and best practices.
    """
    print("\n" + "=" * 70)
    print("ADVANCED USAGE EXAMPLES")
    print("=" * 70)

    print("""
1. Comparing Training Dynamics Across Runs:

    # Load checkpoints from different training runs
    comparator = CheckpointComparator(
        checkpoint_paths=[
            'run1/step_5000',
            'run2/step_5000',
            'run3/step_5000'
        ],
        encoder_cls=TokenToWaveEncoder,
        decoder_cls=WaveToTokenDecoder
    )

    # Compare on validation set
    comparison = comparator.compare_on_dataset(
        dataloader=val_loader,
        max_batches=100
    )

    # Plot pairwise divergence
    for metric_name, matrix in comparison['divergences'].items():
        fig = comparator.plot_pairwise_divergence_matrix(
            matrix,
            metric_name=metric_name,
            save_path=f'divergence_{metric_name}.png'
        )

2. Input Clustering for Data Analysis:

    comparator = InputComparator(model)

    # Generate waves for entire dataset
    all_waves = []
    for batch in dataloader:
        waves = comparator.compare_inputs([batch])
        all_waves.extend(waves)

    # Hierarchical clustering
    cluster_result = comparator.cluster_inputs(
        all_waves,
        method='hierarchical',
        n_clusters=10
    )

    # Visualize dendrogram
    fig = comparator.plot_clustering_results(
        cluster_result,
        all_waves
    )

3. Comprehensive Ablation Study:

    ablator = AblationHelper(model)

    # Test all layers individually
    layer_configs = [
        {'type': 'layer', 'layer_idx': i, 'mode': 'identity'}
        for i in range(model.transformer_num_layers)
    ]

    # Test harmonic ranges
    harmonic_configs = []
    for start in range(0, 64, 8):
        harmonic_configs.append({
            'type': 'harmonics',
            'indices': list(range(start, start + 8)),
            'mode': 'zero'
        })

    # Test wave components
    component_configs = [
        {'type': 'wave_component', 'component': c, 'mode': m}
        for c in ['freq', 'amp', 'phase']
        for m in ['zero', 'random', 'mean']
    ]

    # Run comprehensive study
    all_configs = layer_configs + harmonic_configs + component_configs
    results = ablator.run_ablation_study(
        ablation_configs=all_configs,
        dataloader=val_loader
    )

    # Save results
    results.to_csv('comprehensive_ablation.csv')

    # Visualize
    fig = ablator.plot_ablation_results(results)

4. Custom Evaluation Function:

    def custom_eval_fn(model, dataloader):
        '''Custom evaluation with multiple metrics'''
        model.eval()

        metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'perplexity': 0.0,
            'custom_metric': 0.0
        }

        with torch.no_grad():
            for batch in dataloader:
                # Your evaluation logic here
                pass

        return metrics

    results = ablator.run_ablation_study(
        ablation_configs=configs,
        eval_fn=custom_eval_fn,
        dataloader=val_loader
    )

    """)


def main():
    """
    Main demonstration function.
    """
    print("\n" + "=" * 70)
    print("WAVE TRANSFORMER - COMPARATIVE ANALYSIS TOOLS DEMONSTRATION")
    print("=" * 70)

    print("""
This script demonstrates the three comparative analysis tools:

1. CheckpointComparator - Compare training checkpoints
   - Measure divergence between checkpoints
   - Identify critical training points
   - Analyze layer-wise evolution

2. InputComparator - Compare input representations
   - Compute similarity matrices
   - Cluster similar inputs
   - Visualize with t-SNE/UMAP

3. AblationHelper - Systematic ablation studies
   - Ablate harmonics, layers, or components
   - Run comprehensive studies
   - Visualize component importance

Note: Most code is commented out as it requires actual trained models
and data. Uncomment and adjust paths to run with your setup.
    """)

    # Run demos
    demo_checkpoint_comparison()
    demo_input_comparison()
    demo_ablation_study()
    demonstrate_advanced_usage()

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("""
For detailed documentation, see:
- src/wave_transformer/analysis/comparative/README.md
- COMPARATIVE_ANALYSIS_IMPLEMENTATION.md

To run these tools:
1. Train a Wave Transformer model
2. Save checkpoints during training
3. Uncomment and adjust the code above
4. Run this script or integrate into your workflow

For help:
    from wave_transformer.analysis import CheckpointComparator
    help(CheckpointComparator)
    """)


if __name__ == "__main__":
    main()
