import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_scheduler
from tqdm import tqdm

from wave_transformer.audio.audio_dataset import VCTKDataset, VCTKCollator
from wave_transformer.audio.audio_wave_encoder import AudioToWave
from wave_transformer.core.transformer import WaveTransformer
from wave_transformer.language_modelling.token_decoder import WaveToTokenDecoder


# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = "./checkpoints_tts"
os.makedirs(save_dir, exist_ok=True)

num_epochs = 5
batch_size = 16
eval_every = 1000   # steps
save_every = 2000   # steps
lr = 5e-4


# --- Components ---
audio_encoder = AudioToWave(
    num_harmonics=64,
    sample_rate=24000,
    learnable_filterbank=False
).to(device)

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
vocab_size = tokenizer.vocab_size

wave_decoder = WaveToTokenDecoder(vocab_size, num_heads=8,num_heads_kv=8).to(device)

model = WaveTransformer(
    wave_encoder=audio_encoder,
    wave_decoder=wave_decoder
).to(device)

print("Model:\n", model)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
# --- Dataset ---
dataset = VCTKDataset(
    "./VCTK-Corpus-0.92",
    sample_rate=24000,
    max_len_sec=4,
    return_text=True,
    file_format="flac",
    wav_folder="wav48_silence_trimmed"
)

# Train/val split
val_size = int(0.025 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

collator = VCTKCollator(
    tokenizer=tokenizer,
    return_text=True,
    device=device
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

# --- Optimizer + scheduler ---
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = get_scheduler("linear", optimizer, num_warmup_steps=500, num_training_steps=num_epochs * len(train_loader))

ctc_loss_fn = torch.nn.CTCLoss(
    blank=tokenizer.pad_token_id,
    zero_infinity=True
)


# --- Evaluation ---
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    count = 0
    for batch in tqdm(val_loader, desc="Evaluating", leave=False):
        waveforms = batch["waveforms"].to(device)
        input_ids = batch["input_ids"].to(device)

        logits = model({"audio": waveforms})  # [B, T, V]
        log_probs = F.log_softmax(logits, dim=-1)

        # Prepare CTC inputs
        input_lengths = torch.full(
            (logits.size(0),),
            logits.size(1),
            dtype=torch.long,
            device=device
        )

        # Calculate actual label lengths
        label_lengths = []
        targets = []
        for ids in input_ids:
            if torch.is_tensor(ids):
                length = (ids != tokenizer.pad_token_id).sum().item()
                valid_ids = ids[ids != tokenizer.pad_token_id].tolist()
            else:
                length = len(ids)
                valid_ids = ids
            label_lengths.append(length)
            targets.extend(valid_ids)

        label_lengths = torch.tensor(label_lengths, dtype=torch.long, device=device)
        targets = torch.tensor(targets, dtype=torch.long, device=device)

        # CTC loss
        loss = ctc_loss_fn(
            log_probs.transpose(0, 1),  # (T, B, V)
            targets,
            input_lengths,
            label_lengths
        )
        total_loss += loss.item()
        count += 1

    return total_loss / max(1, count)


# --- Training loop ---
global_step = 0
for epoch in range(num_epochs):
    model.train()
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in progress:
        waveforms = batch["waveforms"].squeeze(1).to(device)  # [B, T]
        input_ids = batch["input_ids"]  # Keep on CPU for now

        # Forward pass
        logits = model({"audio": waveforms})  # Check if model needs dict or tensor
        log_probs = F.log_softmax(logits, dim=-1)

        # Prepare CTC inputs
        input_lengths = torch.full(
            (logits.size(0),),
            logits.size(1),
            dtype=torch.long,
            device=device
        )

        # Calculate actual label lengths
        label_lengths = []
        targets = []
        for ids in input_ids:
            if torch.is_tensor(ids):
                length = (ids != tokenizer.pad_token_id).sum().item()
                valid_ids = ids[ids != tokenizer.pad_token_id].tolist()
            else:
                length = len(ids)
                valid_ids = ids
            label_lengths.append(length)
            targets.extend(valid_ids)

        label_lengths = torch.tensor(label_lengths, dtype=torch.long, device=device)
        targets = torch.tensor(targets, dtype=torch.long, device=device)

        # CTC loss
        loss = ctc_loss_fn(
            log_probs.transpose(0, 1),  # (T, B, V)
            targets,
            input_lengths,
            label_lengths
        )

        # Check for invalid loss
        if not torch.isfinite(loss):
            print(f"Warning: Invalid loss at step {global_step}, skipping...")
            continue

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        global_step += 1
        progress.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

        # Evaluation
        if global_step % eval_every == 0:

            val_loss = evaluate(model, val_loader)
            print(f"\nStep {global_step}: val_loss={val_loss:.4f}")

        # Save checkpoint
        if global_step % save_every == 0:
            save_path = os.path.join(save_dir, f"model_step{global_step}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": global_step,
                "epoch": epoch
            }, save_path)
            print(f"✅ Saved checkpoint to {save_path}")
