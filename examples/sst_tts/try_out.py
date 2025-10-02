import torch
import torchaudio
import torch.nn.functional as F
from transformers import AutoTokenizer

from wave_transformer.audio.audio_wave_encoder import AudioToWave
from wave_transformer.core.transformer import WaveTransformer
from wave_transformer.language_modelling.token_decoder import WaveToTokenDecoder


# --- Config ---
checkpoint_path = "./checkpoints_tts/model_step2000.pt"
tokenizer_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# --- Build model ---
audio_encoder = AudioToWave(
    num_harmonics=64,
    sample_rate=24000,
    learnable_filterbank=False
).to(device)

wave_decoder = WaveToTokenDecoder(tokenizer.vocab_size).to(device)

model = WaveTransformer(
    wave_encoder=audio_encoder,
    wave_decoder=wave_decoder
).to(device)

# --- Load checkpoint ---
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"✅ Loaded checkpoint from {checkpoint_path}")


# --- Preprocess audio ---
def load_audio(file_path, target_sr=24000, max_len_sec=4):
    wav, sr = torchaudio.load(file_path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.mean(dim=0, keepdim=True)  # mono
    wav = wav / (wav.abs().max() + 1e-8)  # normalize

    # pad/truncate
    max_len = target_sr * max_len_sec
    if wav.size(1) > max_len:
        wav = wav[:, :max_len]
    else:
        wav = torch.nn.functional.pad(wav, (0, max_len - wav.size(1)))

    return wav.unsqueeze(0)  # [1, 1, T]


# --- Greedy decode ---
def greedy_decode(logits):
    # logits: [B, T, V]
    preds = torch.argmax(logits, dim=-1)  # [B, T]
    tokens = []
    for seq in preds:
        last = None
        seq_tokens = []
        for idx in seq.tolist():
            if idx != last and idx != tokenizer.pad_token_id:  # collapse repeats + ignore pad
                seq_tokens.append(idx)
            last = idx
        tokens.append(seq_tokens)
    return tokens


# --- Run inference ---
def transcribe(file_path):
    wav = load_audio(file_path).to(device)
    with torch.no_grad():
        logits = model({"audio": wav})  # [1, T, V]
        log_probs = F.log_softmax(logits, dim=-1)
        tokens = greedy_decode(log_probs)

    decoded = tokenizer.decode(tokens[0])
    return decoded


if __name__ == "__main__":
    test_file = "./VCTK-Corpus-0.92/wav48_silence_trimmed/p227/p227_001_mic1.flac"
    result = transcribe(test_file)
    print("🎤 Transcription:", result)
