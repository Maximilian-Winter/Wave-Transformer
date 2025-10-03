import torch
from tokenizers import Tokenizer

from wave_transformer.language_modelling.train_utils import generate_text, load_model_bundle, test_generation

model, _ = load_model_bundle(
    load_dir="pre-train/results",
    prefix="wave_transformer_batch_10000",
    epoch=1,
    map_location="cuda"
)
tokenizer = Tokenizer.from_file("pre-train/SmolLM2-135M-Instruct-Tokenizer.json")
pad_token_id = tokenizer.token_to_id("<|im_end|>") or 0

print("Pad Token ID:", pad_token_id)
print("Pad Token:", tokenizer.decode([pad_token_id], False))
test_generation(model=model.to(device="cuda"), tokenizer=tokenizer, device="cuda")
gen = generate_text(model=model.to(device="cuda"), prompt="The tao", tokenizer=tokenizer, device="cuda")
print("Generate:", gen)

