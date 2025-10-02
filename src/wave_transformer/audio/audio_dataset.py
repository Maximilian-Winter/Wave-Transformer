import os
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset
import glob

class FLACDataset(Dataset):
    def __init__(self, root_dir, sample_rate=24000, transform=None):
        self.files = glob.glob(os.path.join(root_dir, "**/*.flac"), recursive=True)
        self.sample_rate = sample_rate
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        waveform, sr = torchaudio.load(path)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        if self.transform:
            waveform = self.transform(waveform)

        return {"waveform": waveform, "path": path}

class VCTKAudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=24000, max_len_sec=4, file_format: str = "wav", wav_folder: str = "wav48"):
        self.files = []
        self.sample_rate = sample_rate
        self.max_len = sample_rate * max_len_sec

        for speaker in os.listdir(root_dir):
            spk_dir = os.path.join(root_dir, speaker, wav_folder)
            if os.path.isdir(spk_dir):
                for f in os.listdir(spk_dir):
                    if f.endswith(f"./{file_format}"):
                        self.files.append(os.path.join(spk_dir, f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        wav, sr = torchaudio.load(path)

        # Resample + mono
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.mean(dim=0, keepdim=True)  # mono

        # Normalize
        wav = wav / wav.abs().max()

        # Truncate or pad
        if wav.size(1) > self.max_len:
            wav = wav[:, :self.max_len]
        else:
            pad_len = self.max_len - wav.size(1)
            wav = torch.nn.functional.pad(wav, (0, pad_len))

        return wav


class VCTKDataset(Dataset):
    def __init__(
        self,
        root_dir,
        sample_rate=24000,
        max_len_sec=4,
        return_text=True,
        file_format: str = "flac",  # or "wav"
        wav_folder: str = "wav48_silence_trimmed",
        txt_folder: str = "txt"
    ):
        """
        Args:
            root_dir (str): Path to the VCTK dataset root (contains wav48_silence_trimmed/ and txt/).
            sample_rate (int): Desired sample rate (default: 24kHz).
            max_len_sec (int): Clip length in seconds (waveform is padded/truncated).
            return_text (bool): Whether to also return transcript text.
        """
        self.sample_rate = sample_rate
        self.max_len = sample_rate * max_len_sec
        self.return_text = return_text

        self.data = []
        wav_dir = os.path.join(root_dir, wav_folder)
        txt_dir = os.path.join(root_dir, txt_folder)

        # Collect files + match transcripts
        for speaker in sorted(os.listdir(wav_dir)):
            spk_wav_dir = os.path.join(wav_dir, speaker)
            spk_txt_dir = os.path.join(txt_dir, speaker)
            if not os.path.isdir(spk_wav_dir):
                continue

            for fname in sorted(os.listdir(spk_wav_dir)):
                if fname.endswith(f".{file_format}"):
                    wav_path = os.path.join(spk_wav_dir, fname)

                    # remove _mic* suffix for matching transcript
                    base_id = fname.split("_mic")[0]
                    txt_fname = base_id + ".txt"
                    txt_path = os.path.join(spk_txt_dir, txt_fname)

                    if self.return_text:
                        if os.path.exists(txt_path):
                            with open(txt_path, "r", encoding="utf-8") as f:
                                transcript = f.read().strip()
                            # only append if transcript is not empty
                            if transcript:
                                self.data.append((wav_path, transcript, speaker))
                        else:
                            # skip samples with no transcript
                            continue
                    else:
                        self.data.append((wav_path, None, speaker))

        print(f"Found {len(self.data)} samples in VCTK ({file_format}).")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav_path, transcript, speaker = self.data[idx]

        # Load audio
        wav, sr = torchaudio.load(wav_path)

        # Resample + mono
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.mean(dim=0, keepdim=True)  # mono

        # Normalize
        wav = wav / (wav.abs().max() + 1e-8)

        # Truncate or pad
        if wav.size(1) > self.max_len:
            wav = wav[:, :self.max_len]
        else:
            pad_len = self.max_len - wav.size(1)
            wav = torch.nn.functional.pad(wav, (0, pad_len))

        return {
            "waveform": wav,
            "text": transcript,
            "speaker": speaker,
            "path": wav_path
        }




class VCTKCollator:
    def __init__(self, tokenizer=None, return_text=True, device="cpu"):
        """
        Args:
            tokenizer: Hugging Face tokenizer (optional, for transcripts).
            return_text (bool): Whether to return tokenized text.
            device: Target device for tensors (e.g., "cuda").
        """
        self.tokenizer = tokenizer
        self.return_text = return_text
        self.device = device

    def __call__(self, batch):
        """
        Args:
            batch: List of (wav, transcript, speaker) tuples
        Returns:
            dict with 'waveforms', 'input_ids', 'attention_mask', 'speakers'
        """
        transcripts = None
        if self.return_text:
            transcripts = [item["text"].strip() for item in batch if item["text"]]

        waveforms = [item["waveform"].to(self.device) for item in batch]
        speakers = [item["speaker"] for item in batch]

        # Stack waveforms (already padded in dataset)
        waveforms = torch.stack(waveforms, dim=0)

        result = {"waveforms": waveforms, "speakers": speakers}

        if self.return_text and self.tokenizer is not None:
            encoded = self.tokenizer(
                transcripts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            result["input_ids"] = encoded["input_ids"].to(self.device)
            result["attention_mask"] = encoded["attention_mask"].to(self.device)
            result["transcripts"] = transcripts  # keep raw text for inspection

        return result


class VCTKCollatorSpeakerEmbedding:
    def __init__(self, tokenizer=None, return_text=True, device="cpu", speaker2id=None):
        """
        Args:
            tokenizer: Hugging Face tokenizer (optional, for transcripts).
            return_text (bool): Whether to return tokenized text.
            device: Target device for tensors (e.g., "cuda").
            speaker2id (dict): Mapping {speaker_name: speaker_id}.
                               If None, will build on the fly.
        """
        self.tokenizer = tokenizer
        self.return_text = return_text
        self.device = device
        self.speaker2id = speaker2id or {}
        self.next_speaker_id = max(self.speaker2id.values(), default=-1) + 1

    def _get_speaker_id(self, speaker_name):
        """Map speaker string (e.g. 'p225') to numeric ID."""
        if speaker_name not in self.speaker2id:
            self.speaker2id[speaker_name] = self.next_speaker_id
            self.next_speaker_id += 1
        return self.speaker2id[speaker_name]

    def __call__(self, batch):
        """
        Args:
            batch: List of (wav, transcript, speaker_name) tuples
        Returns:
            dict with 'waveforms', 'input_ids', 'attention_mask', 'speakers'
        """
        waveforms = [item[0].to(self.device) for item in batch]
        transcripts = [item[1] for item in batch]
        speakers = [item[2] for item in batch]

        # Stack waveforms
        waveforms = torch.stack(waveforms, dim=0)

        # Convert speaker IDs
        speaker_ids = torch.tensor(
            [self._get_speaker_id(s) for s in speakers],
            dtype=torch.long, device=self.device
        )

        result = {"waveforms": waveforms, "speakers": speaker_ids}

        if self.return_text and self.tokenizer is not None:
            encoded = self.tokenizer(
                transcripts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            result["input_ids"] = encoded["input_ids"].to(self.device)
            result["attention_mask"] = encoded["attention_mask"].to(self.device)
            result["transcripts"] = transcripts  # keep raw text

        return result


if __name__ == "__main__":
    dataset = VCTKDataset("/path/to/VCTK-Corpus", sample_rate=24000, max_len_sec=4, return_text=True)
    wav, text, spk = dataset[0]

    print(wav.shape)  # torch.Size([1, 96000]) for 4s at 24kHz
    print(text)  # transcript string
    print(spk)  # speaker ID like "p225"

    ##############

    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    # Load dataset
    dataset = VCTKDataset("/path/to/VCTK-Corpus", sample_rate=24000, max_len_sec=4, return_text=True)

    # Tokenizer for text conditioning
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Collator
    collator = VCTKCollator(tokenizer=tokenizer, return_text=True, device="cuda")

    # DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collator)

    batch = next(iter(loader))
    print(batch["waveforms"].shape)  # torch.Size([8, 1, 96000])
    print(batch["input_ids"].shape)  # torch.Size([8, seq_len])
    print(batch["speakers"])  # list of speaker IDs

    #########

    # Initialize collator
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    collator = VCTKCollatorSpeakerEmbedding(tokenizer=tokenizer, return_text=True, device="cuda")

    # DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collator)

    batch = next(iter(loader))
    print(batch["waveforms"].shape)  # [8, 1, 96000]
    print(batch["speakers"].shape)   # [8], integer IDs
    print(batch["input_ids"].shape)  # [8, seq_len]

    # Example speaker embedding layer
    speaker_emb = nn.Embedding(len(collator.speaker2id), 128).to("cuda")
    speaker_vecs = speaker_emb(batch["speakers"])
    print(speaker_vecs.shape)  # [8, 128]
