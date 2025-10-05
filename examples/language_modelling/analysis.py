import copy

from tokenizers import processors, Tokenizer

from wave_transformer.analysis.generation import (
    LiveGenerationVisualizer,
    WaveTrajectoryTracker,
    GenerationConfidenceTracker,
    RoundTripAnalyzer
)
from wave_transformer.core.signal_processor import SignalTransformer

seq_len = 512
model_name = "./pre-train/SmolLM2-135M-Instruct-Tokenizer.json"
train_tokenizer = Tokenizer.from_file(model_name)

train_tokenizer.add_special_tokens(["<|bos|>", "<|eos|>", "<|pad|>"])

bos_token_id = train_tokenizer.token_to_id("<|bos|>")
eos_token_id = train_tokenizer.token_to_id("<|eos|>")
pad_token_id = train_tokenizer.token_to_id("<|pad|>")

tokenizer = copy.deepcopy(train_tokenizer)

tokenizer.post_processor = processors.TemplateProcessing(
    single="<|bos|> $A",
    pair="<|bos|> $A <|bos|> $B",
    special_tokens=[
        ("<|bos|>", bos_token_id)
    ],
)

prompts = [
    "The tao that can be told",
    "Success is as dangerous as failure."
]
model = SignalTransformer.load("./pre-train/results_signal/epoch_4_final").to(device="cuda")
# Initialize all analyzers
visualizer = LiveGenerationVisualizer(model, tokenizer, device="cuda")
trajectory_tracker = WaveTrajectoryTracker()
confidence_tracker = GenerationConfidenceTracker(k=10)
roundtrip_analyzer = RoundTripAnalyzer(model)

# Generate with live visualization
output_ids, waves = visualizer.generate_with_visualization(
    prompt="The tao that can be told",
    max_length=50,
    temperature=0.8,
    interactive=False  # Set to True for real-time display
)

# Track trajectory using stored waves
for step, wave in enumerate(waves):
    trajectory_tracker.track_step(step, wave)

    # Also track confidence (need logits - get from model)
    logits, _ = model(
        encoder_input={'token_ids': output_ids[:, :step+1]},
        return_encoder_outputs=True
    )
    next_token = output_ids[0, step+1] if step+1 < output_ids.size(1) else output_ids[0, -1]
    confidence_tracker.track_step(step, logits[:, -1, :], next_token.item(), wave)

# Generate all visualizations
visualizer.create_animation("generation_anim.mp4", fps=2)
trajectory_tracker.plot_trajectory(save_path="trajectory.png")
confidence_tracker.plot_confidence_trajectory(save_path="confidence.png")
confidence_tracker.plot_wave_confidence_correlation(save_path="correlation.png")

# Check for issues
collapse = trajectory_tracker.detect_mode_collapse()
if collapse['mode_collapse_detected']:
    print("⚠️ Mode collapse detected!")

uncertain = confidence_tracker.identify_uncertain_regions(threshold=0.5)
print(f"Found {len(uncertain)} uncertain tokens")

# Round-trip analysis on generated sequence
roundtrip_result = roundtrip_analyzer.analyze_roundtrip(output_ids)
print(f"Round-trip accuracy: {roundtrip_result.reconstruction_accuracy:.2%}")
roundtrip_analyzer.plot_roundtrip_analysis(roundtrip_result, tokenizer)

# Export all data
trajectory_tracker.export_trajectory("trajectory_data.csv")
gen_summary = visualizer.get_generation_summary()
conf_summary = confidence_tracker.get_confidence_summary()
traj_summary = trajectory_tracker.get_trajectory_statistics()

print("\n=== Generation Analysis Summary ===")
print(f"Tokens generated: {gen_summary['num_tokens_generated']}")
print(f"Average confidence: {conf_summary['max_probability']['mean']:.3f}")
print(f"Average entropy: {conf_summary['entropy']['mean']:.3f}")
print(f"Energy trend: {traj_summary['energy']['trend']:.6f}")
print(f"Round-trip accuracy: {roundtrip_result.reconstruction_accuracy:.2%}")