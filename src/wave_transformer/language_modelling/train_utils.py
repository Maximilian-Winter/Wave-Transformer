import json
from pathlib import Path
import math as _math

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from wave_transformer.core.wave_transformer import WaveTransformer
from wave_transformer.language_modelling.token_decoder import WaveToTokenDecoder
from wave_transformer.language_modelling.token_encoder import TokenToWaveEncoder


def compute_distillation_loss(student_logits, teacher_logits, targets, pad_token_id, alpha=0.5, temperature=2.0):
    """
    Compute combined loss for knowledge distillation.

    Args:
        student_logits: [batch, seq_len, vocab] from student model
        teacher_logits: [batch, seq_len, vocab] from teacher model
        targets: ground truth token ids [batch, seq_len]
        pad_token_id: id of the padding token
        alpha: weight for distillation loss vs CE
        temperature: softening factor for teacher/student distributions
    """
    vocab_size = student_logits.size(-1)

    # --- Standard LM loss ---
    ce_loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    lm_loss = ce_loss_fct(
        student_logits.view(-1, vocab_size),
        targets.view(-1)
    )

    # --- Distillation KL loss ---
    # soften both student & teacher distributions
    log_probs_student = nn.functional.log_softmax(student_logits / temperature, dim=-1)
    probs_teacher = nn.functional.softmax(teacher_logits / temperature, dim=-1)

    # KL divergence
    kl_loss = nn.functional.kl_div(
        log_probs_student, probs_teacher,
        reduction="batchmean"
    ) * (temperature ** 2)

    # Combine
    loss = alpha * lm_loss + (1 - alpha) * kl_loss
    return loss, lm_loss.item(), kl_loss.item()


@torch.no_grad()
def generate_text(model, tokenizer, prompt, device, max_tokens=100,
                  temperature=0.75, top_k=0, top_p=0.9, min_p=0.0, repetition_penalty=1.2, max_seq_length=128):
    model.eval()

    tokenized = tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
    generated = torch.tensor(tokenized.ids, dtype=torch.long, device=device).unsqueeze(0)
    attn = torch.tensor(tokenized.attention_mask, dtype=torch.bool, device=device).unsqueeze(0)

    # ✅ Detect EOS token
    eos_token_id = tokenizer.token_to_id("<|eos|>")

    for _ in range(max_tokens):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if not isinstance(model, WaveTransformer) and not isinstance(model, PyramidWaveTransformer):
                logits = model(generated, attention_mask=attn)
            else:
                logits = model({"token_ids": generated}, attention_mask=attn)
            logits = get_logits_tensor(logits)
            next_logits = logits[:, -1, :].squeeze(0)


        # ✅ Proper repetition penalty
        if repetition_penalty != 1.0 and generated.numel() > 0:
            for token_id in set(generated[0].tolist()):
                logit_val = next_logits[token_id]
                if logit_val < 0:
                    next_logits[token_id] *= repetition_penalty
                else:
                    next_logits[token_id] /= repetition_penalty

        # Temperature scaling
        if temperature != 1.0:
            next_logits = next_logits / max(1e-6, temperature)

        probabilities = torch.softmax(next_logits, dim=-1)

        # Min-p filtering
        if min_p > 0.0:
            threshold = probabilities.max() * min_p
            probabilities = torch.where(probabilities >= threshold, probabilities, torch.zeros_like(probabilities))
            sum_probs = probabilities.sum()
            if sum_probs.item() > 0:
                probabilities = probabilities / sum_probs
            else:
                probabilities = torch.softmax(next_logits, dim=-1)

        # Top-k filtering
        if top_k and top_k > 0:
            k = min(top_k, probabilities.size(-1))
            top_values, top_indices = torch.topk(probabilities, k)
            filtered = torch.zeros_like(probabilities)
            filtered[top_indices] = top_values
            probabilities = filtered / filtered.sum()

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff_idx = torch.searchsorted(cumulative, torch.tensor(top_p, device=device)).item() + 1
            keep_indices = sorted_indices[:cutoff_idx]
            filtered = torch.zeros_like(probabilities)
            filtered[keep_indices] = probabilities[keep_indices]
            probabilities = filtered / filtered.sum()

        # Handle numerical stability
        probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
        probabilities = probabilities + 1e-10
        probabilities = probabilities / probabilities.sum()

        next_token = torch.multinomial(probabilities, 1).item()

        # ✅ Stop if EOS token generated
        if eos_token_id is not None and next_token == eos_token_id:
            break
        if generated.shape[1] + 1 >= max_seq_length:
            break
        generated = torch.cat([generated, torch.tensor([[next_token]], device=device, dtype=torch.long)], dim=1)
        attn = torch.cat([attn, torch.tensor([[1]], device=device, dtype=torch.bool)], dim=1)

    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


def test_generation(model, tokenizer, max_tokens=50, device="cuda", temperature=0.7, top_p=0.9, repetition_penalty=1.2, prompts=None):
    # Generate samples
    if prompts is None:
        prompts = [
            "The tao that can be told",
            "Success is as dangerous as failure.",
            "Major Premise: All matter is composed of atoms,",
            "Claim: The most informative and foundational concept in science,",
            "Claim: A string with both ends fixed can only oscillate"
        ]

    generations = []
    for prompt in prompts:
        text = generate_text(
            model, tokenizer,
            prompt, device, max_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty,
        )
        print(f"\n\nInput: '{prompt}'\nResult: '{text}'")
        generations.append(text)
    return generations


def camel_to_snake(camel_case):
    """Convert CamelCase string to snake_case."""
    result = []
    for i, char in enumerate(camel_case):
        if char.isupper() and i > 0:
            result.append('_')
        result.append(char.lower())
    return ''.join(result)


def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, base_lr, final_lr=0.0):
    """Linear warmup to base_lr, then cosine decay to final_lr."""
    def lr_lambda(step):
        # step is the *optimizer* step (after grad accumulation)
        if step < warmup_steps:
            return step / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # cosine from 1.0 -> (final_lr/base_lr)
        cos = 0.5 * (1.0 + _math.cos(_math.pi * progress))
        return (final_lr / base_lr) + (1 - final_lr / base_lr) * cos
    return LambdaLR(optimizer, lr_lambda)



def save_training_chronicle(chronicle, result_dir, experiment_name, timestamp):
    output_path = Path(f"{result_dir}/{experiment_name}_{timestamp.replace(':', '-')}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chronicle, f, indent=2, default=str)
    return output_path


from pathlib import Path
from typing import Tuple, Optional, Union
import torch


def save_model_bundle(
        model: WaveTransformer,
        save_dir: Union[str, Path],
        epoch: int = None,
        global_step: int = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,

):
    """
    Save model, encoder, decoder, and optional training state.

    Args:
        model: WaveTransformer instance
        save_dir: Directory to save files
        prefix: Prefix for saved files
        epoch: Optional epoch number to include in filename
        optimizer: Optional optimizer state to save
        scheduler: Optional scheduler state to save
    """
    if not isinstance(model, WaveTransformer):
        return None
    model.save(save_dir)

    if optimizer is not None or scheduler is not None:
        training_state = {}
        if epoch is not None:
            training_state['epoch'] = epoch
        if global_step is not None:
            training_state['global_step'] = global_step
        if optimizer is not None:
            training_state['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            training_state['scheduler_state_dict'] = scheduler.state_dict()
        save_dir = Path(save_dir)
        torch.save(training_state, save_dir / f"training.pt")

    print(f"✓ Saved model bundle to {save_dir}")


def load_model_bundle(
        load_dir: Union[str, Path],
        map_location: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Tuple[WaveTransformer, dict]:
    """
    Load model, encoder, decoder, and optional training state.

    Args:
        load_dir: Directory containing saved files
        map_location: Device to load model to ('cpu', 'cuda', etc.)
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into

    Returns:
        Tuple of (WaveTransformer model, training_state dict)
    """


    # Load model
    model = WaveTransformer.load(
        load_dir,
        encoder_cls=TokenToWaveEncoder,
        decoder_cls=WaveToTokenDecoder,
        map_location=map_location
    )
    load_dir = Path(load_dir)
    # Load training state if exists
    training_state = {}
    training_path = load_dir / f"training.pt"
    if training_path.exists():
        training_state = torch.load(training_path, map_location=map_location)

        if optimizer is not None and 'optimizer_state_dict' in training_state:
            optimizer.load_state_dict(training_state['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in training_state:
            scheduler.load_state_dict(training_state['scheduler_state_dict'])

    print(f"✓ Loaded model bundle from {load_dir}")

    return model, training_state

def extract_architecture_details(model):
    layer_details = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            layer_details.append({
                'name': name,
                'type': module.__class__.__name__,
                'parameters': sum(p.numel() for p in module.parameters())
            })

    return {
        'representation': str(model),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'layer_details': layer_details
    }



def prepare_autoregressive_batch(batch, pad_token_id):
    token_ids = batch['input_ids']
    attention_mask = batch['attention_mask']

    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.bool()

    inputs = token_ids[:, :-1]
    targets = token_ids[:, 1:]
    input_mask = attention_mask[:, :-1]

    return inputs, targets, input_mask

def get_logits_tensor(logits):
    if isinstance(logits, tuple):
        return logits[0]
    elif isinstance(logits, torch.Tensor):
        return logits
    else:
        return logits.logits

def lm_total_loss(logits, targets, pad_token_id):
    ce = compute_language_modeling_loss(logits, targets, pad_token_id, label_smoothing=0.05)
    zl = z_loss(logits, coeff=1e-4)
    #ul = repetition_unlikelihood(logits, targets, pad_token_id, window=64, coeff=2e-2)
    return ce + zl, {"ce": ce.item(), "z": zl.item()}

def compute_language_modeling_loss(logits, targets, pad_token_id, label_smoothing=0.05):
    # logits: (B, T, V), targets: (B, T)
    logits = logits.float()              # avoid bf16 precision issues inside CE
    targets = targets.long()
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        ignore_index=pad_token_id,
        label_smoothing=label_smoothing,
        reduction="mean",
    )
    return loss

def z_loss(logits, coeff=1e-4):
    # logits: (B, T, V)
    # z = logsumexp along vocab
    z = torch.logsumexp(logits.float(), dim=-1)        # (B, T)
    return coeff * (z ** 2).mean()

def repetition_unlikelihood(logits, targets, pad_token_id, window=64, coeff=2e-2):
    """
    Penalize predicting tokens that already appeared in the last `window` steps
    of the same sequence. Cheap token-level variant.
    """
    logits = logits.float()               # (B, T, V)
    B, T, V = logits.shape
    targets = targets.long()              # (B, T)

    # Build a mask of "negative" tokens per position: tokens seen in the last `window` steps
    neg_masks = []
    for b in range(B):
        seen = set()
        row_masks = []
        for t in range(T):
            # tokens seen in [max(0, t-window), t)
            start = max(0, t - window)
            context = targets[b, start:t]
            # exclude PAD
            seen_tokens = set(context[context != pad_token_id].tolist())
            row_masks.append(seen_tokens)
        neg_masks.append(row_masks)

    # Compute p = softmax(logits)
    probs = torch.softmax(logits, dim=-1)              # (B, T, V)
    # For each (b,t), gather probabilities of negatives and apply -log(1 - p)
    ul_terms = []
    for b in range(B):
        for t in range(T):
            neg = neg_masks[b][t]
            if not neg:
                continue
            p_neg = probs[b, t, list(neg)]             # (K,)
            # clamp to avoid log(0)
            ul = -torch.log(torch.clamp(1.0 - p_neg, min=1e-6))
            ul_terms.append(ul.mean())

    if not ul_terms:
        return logits.new_tensor(0.0)
    return coeff * torch.stack(ul_terms).mean()

def compute_diversity_metrics(texts, n=3):
    all_ngrams = []
    repetition_scores = []

    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
        repetition = (len(ngrams) - len(set(ngrams))) / max(1, len(ngrams))
        repetition_scores.append(repetition)

    distinct_ratio = len(set(all_ngrams)) / max(1, len(all_ngrams))
    mean_repetition = sum(repetition_scores) / max(1, len(repetition_scores))

    return {
        "mean_repetition": mean_repetition,
        f"distinct_{n}": distinct_ratio
    }

def diversity_report(texts, ns=(1,2,3)):
    out = {}
    for n in ns:
        all_ngrams, reps = [], []
        for t in texts:
            toks = t.split()  # better: tokenizer tokens if available
            ngrams = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
            if ngrams:
                all_ngrams.extend(ngrams)
                reps.append((len(ngrams) - len(set(ngrams))) / len(ngrams))
        out[f"distinct_{n}"] = len(set(all_ngrams)) / max(1, len(all_ngrams))
        out[f"mean_repetition_{n}"] = sum(reps) / max(1, len(reps))
    return out
