"""
ChatML Tokenizer for Transformer Language Models
Uses HuggingFace's tokenizers library (Rust-backed)
"""

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

def create_chatml_tokenizer(vocab_size=32000):
    """
    Create a BPE tokenizer with ChatML special tokens.
    
    Args:
        vocab_size: Size of the vocabulary
    
    Returns:
        Tokenizer object configured for ChatML format
    """
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.from_file()
    
    # Set up pre-tokenizer (splits on whitespace and punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Set up decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # Define ChatML special tokens
    special_tokens = [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|pad|>",
    ]
    
    # Create trainer with special tokens
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    return tokenizer, trainer, special_tokens


def train_tokenizer(tokenizer, trainer, files):
    """
    Train the tokenizer on your corpus.
    
    Args:
        tokenizer: Tokenizer object
        trainer: Trainer object
        files: List of file paths containing training text
    """
    tokenizer.train(files, trainer)
    return tokenizer


def setup_post_processor(tokenizer, special_tokens):
    """
    Set up post-processor for adding special tokens to sequences.
    """
    # Get token IDs for special tokens
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    im_end_id = tokenizer.token_to_id("<|im_end|>")
    
    # Configure post-processor for ChatML format
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|im_start|> $A <|im_end|>",
        pair="<|im_start|> $A <|im_end|> <|im_start|> $B <|im_end|>",
        special_tokens=[
            ("<|im_start|>", im_start_id),
            ("<|im_end|>", im_end_id),
        ],
    )


def format_chatml_conversation(messages):
    """
    Format a conversation in ChatML style.
    
    Args:
        messages: List of dicts with 'role' and 'content' keys
        
    Returns:
        Formatted string in ChatML format
    
    Example:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    """
    formatted = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        formatted.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    
    return "\n".join(formatted)


if __name__ == "__main__":
    # Example usage
    print("Creating ChatML tokenizer...")
    tokenizer, trainer, special_tokens = create_chatml_tokenizer(vocab_size=32000)
    
    # Train on your corpus (example with dummy file)
    # tokenizer = train_tokenizer(tokenizer, trainer, ["your_corpus.txt"])
    
    # For demo purposes, let's create a small example corpus
    with open("/home/claude/example_corpus.txt", "w") as f:
        f.write("Hello world! This is a test corpus.\n")
        f.write("The quick brown fox jumps over the lazy dog.\n")
        f.write("Machine learning is fascinating.\n")
    
    print("Training tokenizer on example corpus...")
    tokenizer = train_tokenizer(tokenizer, trainer, ["/home/claude/example_corpus.txt"])
    
    # Set up post-processor
    setup_post_processor(tokenizer, special_tokens)
    
    # Save the tokenizer
    tokenizer.save("/home/claude/chatml_tokenizer.json")
    print("Tokenizer saved to chatml_tokenizer.json")
    
    # Example: Format and tokenize a conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI."}
    ]
    
    formatted_text = format_chatml_conversation(messages)
    print("\n" + "="*50)
    print("Formatted ChatML conversation:")
    print("="*50)
    print(formatted_text)
    
    print("\n" + "="*50)
    print("Tokenization example:")
    print("="*50)
    encoding = tokenizer.encode(formatted_text)
    print(f"Tokens: {encoding.tokens[:20]}...")  # Show first 20 tokens
    print(f"IDs: {encoding.ids[:20]}...")
    print(f"Total tokens: {len(encoding.ids)}")
