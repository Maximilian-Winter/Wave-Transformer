import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

from wave_transformer.audio.audio_dataset import VCTKDataset, VCTKCollator
from wave_transformer.audio.audio_wave_decoder import WaveToAudio
from wave_transformer.audio.audio_wave_encoder import AudioToWave
from wave_transformer.core.transformer import WaveTransformer
from wave_transformer.language_modelling.token_decoder import WaveToTokenDecoder
from wave_transformer.language_modelling.token_encoder import TokenToWaveEncoder

# ========================
# Config
# ========================
mode = "sst"  # "sst" = Speech → Text | "tts" = Text → Speech
device = "cuda" if torch.cuda.is_available() else "cpu"

epochs = 5
batch_size = 8
lr = 5e-4
warmup_steps = 500
total_steps = 10000
max_len_sec = 4
sample_rate = 24000

checkpoint_dir = "./checkpoints_tts_sst"
os.makedirs(checkpoint_dir, exist_ok=True)


# ========================
# Tokenizer
# ========================
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
vocab_size = tokenizer.vocab_size


# ========================
# Build Model
# ========================
if mode == "sst":  # Speech-to-Text
    encoder = AudioToWave(num_harmonics=64, sample_rate=sample_rate).to(device)
    decoder = WaveToTokenDecoder(vocab_size).to(device)
    model = WaveTransformer(wave_encoder=encoder, wave_decoder=decoder).to(device)
elif mode == "tts":  # Text-to-Speech
    # Encoder will embed text tokens
    encoder = TokenToWaveEncoder(vocab_size)
    # Decoder generates audio harmonic representation
    decoder = WaveToAudio(num_harmonics=64, sample_rate=sample_rate).to(device)
    model = WaveTransformer(wave_encoder=encoder, wave_decoder=decoder).to(device)
else:
    raise ValueError("Mode must be 'sst' or 'tts'")


# ========================
# Dataset + Loader
# ========================
dataset = VCTKDataset(
    "./VCTK-Corpus-0.92",
    sample_rate=sample_rate,
    max_len_sec=max_len_sec,
    return_text=True,
    file_format="flac",
    wav_folder="wav48_silence_trimmed"
)

collator = VCTKCollator(tokenizer=tokenizer, return_text=True, device=device)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)


# ========================
# Optimizer + Scheduler
# ========================
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = get_scheduler("linear", optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)


# ========================
# Loss Functions
# ========================
ctc_loss_fn = nn.CTCLoss(blank=tokenizer.pad_token_id, zero_infinity=True)
mse_loss_fn = nn.MSELoss()


# ========================
# Training Loop
# ========================
for epoch in range(epochs):
    model.train()
    progress = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
    total_loss = 0

    for step, batch in enumerate(progress):
        optimizer.zero_grad()

        if mode == "sst":  # Speech → Text
            waveforms = batch["waveforms"].to(device)       # [B, 1, T]
            input_ids = batch["input_ids"].to(device)       # [B, L]

            logits = model({"audio": waveforms})            # [B, T, V]
            log_probs = F.log_softmax(logits, dim=-1)

            input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
            label_lengths = torch.tensor([len(ids) for ids in input_ids], dtype=torch.long)

            targets = torch.cat([ids for ids in input_ids])  # flatten labels
            loss = ctc_loss_fn(log_probs.transpose(0, 1), targets, input_lengths, label_lengths)

        elif mode == "tts":  # Text → Speech
            input_ids = batch["input_ids"].to(device)       # [B, L]
            waveforms = batch["waveforms"].to(device)       # [B, 1, T]

            predicted_wave = model({"text": input_ids})     # [B, 1, T]
            loss = mse_loss_fn(predicted_wave, waveforms)   # regression loss on waveform

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress.set_postfix({"loss": loss.item()})

        if step % 1000 == 0 and step > 0:
            ckpt_path = os.path.join(checkpoint_dir, f"{mode}_epoch{epoch+1}_step{step}.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, ckpt_path)
            print(f"💾 Saved checkpoint {ckpt_path}")

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} finished | avg_loss = {avg_loss:.4f}")

    # Save per epoch
    ckpt_path = os.path.join(checkpoint_dir, f"{mode}_epoch{epoch+1}.pt")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, ckpt_path)
    print(f"💾 Saved checkpoint {ckpt_path}")
