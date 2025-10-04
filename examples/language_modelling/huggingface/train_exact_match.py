"""
WaveTransformer training script with HuggingFace Trainer
"""

import json
import random
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, default_data_collator
from datasets import Dataset
from wave_transformer.huggingface.hf_wave_transformer import WaveTransformerConfig, WaveTransformerForCausalLM
#from wave_transformer.language_modelling.train_utils import EffectiveLossCallback, GenerationCallback


def main():
    # Load and prepare data
    with open("dao_de_jing.json", "r", encoding="utf-8") as file:
        chapters = json.load(file)

    texts = chapters * 50

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Model configuration
    config = WaveTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        num_layers=48,
        num_heads=8,
        num_harmonics=64,
        max_seq_len=512,
        dropout=0.1,
        encoder_d_model=512,
        encoder_hidden_mult=2.0,
        encoder_num_heads=8,
        encoder_num_layers=3,
        decoder_d_model=512,
        decoder_hidden_mult=1.75,
        decoder_num_heads=8,
        decoder_num_layers=3,
        decoder_low_rank_output=512,
        use_flash=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = WaveTransformerForCausalLM(config)
    print(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Prepare dataset
    dataset = Dataset.from_list(texts)

    def tokenize_and_prepare(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        out["labels"] = out["input_ids"].copy()
        # Mask out pad tokens in labels
        out["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in seq]
            for seq in out["labels"]
        ]
        return out

    tokenized = dataset.map(
        tokenize_and_prepare,
        batched=True,
        remove_columns=["text"],
        num_proc=3
    )

    # Training arguments
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    args = TrainingArguments(
        output_dir=f"./wave_transformer_model_{timestamp}",  # <-- timestamped folder
        run_name=f"wave_transformer_dao_{timestamp}",  # also tag run name
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,  # safer than 5e-4, less risk of NaNs
        weight_decay=0.05,  # lower, prevents over-regularizing
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        warmup_ratio=0.25,
        num_train_epochs=2,
        lr_scheduler_type="cosine",  # optional: periodic cosine annealing
        bf16=torch.cuda.is_available(),  # efficient if GPU supports it
        tf32=True,
        #dataloader_num_workers=2,
        #dataloader_pin_memory=True,
        logging_steps=100,  # more frequent logging
        save_strategy="epoch",  # save each epoch
        save_total_limit=10,  # keep last 3 checkpoints
        report_to=[],
        push_to_hub=False,
        #load_best_model_at_end=True,  # keep best checkpoint
        do_eval=False,  # no eval dataset right now
        #gradient_checkpointing=True,  # saves VRAM, tiny slowdown
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",  # fused is faster
    )

    # Train model
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=default_data_collator,
        callbacks=[
            #EffectiveLossCallback(args.gradient_accumulation_steps),
            #GenerationCallback(tokenizer=tokenizer, prompts=["The tao that can be told"], generation_loss_threshold=3.6)
        ]
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model("./wave_transformer_final")
    print("Training complete. Model saved to ./wave_transformer_final")

if __name__ == "__main__":
    main()