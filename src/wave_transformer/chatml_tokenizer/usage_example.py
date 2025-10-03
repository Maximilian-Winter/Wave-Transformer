"""
Example usage of the ChatML tokenizer
"""

from tokenizers import Tokenizer

def load_tokenizer(path):
    """Load a saved tokenizer."""
    return Tokenizer.from_file(path)


def encode_chatml_message(tokenizer, role, content):
    """Encode a single ChatML message."""
    message = f"<|im_start|>{role}\n{content}<|im_end|>"
    return tokenizer.encode(message)


def decode_tokens(tokenizer, token_ids):
    """Decode token IDs back to text."""
    return tokenizer.decode(token_ids)


if __name__ == "__main__":
    # Load the tokenizer
    tokenizer = load_tokenizer("/home/claude/chatml_tokenizer.json")
    
    print("ChatML Tokenizer Usage Examples")
    print("="*50)
    
    # Example 1: Encode a single message
    print("\n1. Encoding a single message:")
    encoding = encode_chatml_message(tokenizer, "user", "Hello, how are you?")
    print(f"   Text: 'Hello, how are you?'")
    print(f"   Tokens: {encoding.tokens}")
    print(f"   IDs: {encoding.ids}")
    
    # Example 2: Decode back to text
    print("\n2. Decoding tokens back to text:")
    decoded = decode_tokens(tokenizer, encoding.ids)
    print(f"   Decoded: {decoded}")
    
    # Example 3: Full conversation
    print("\n3. Full conversation encoding:")
    conversation = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
2+2 equals 4.<|im_end|>"""
    
    encoding = tokenizer.encode(conversation)
    print(f"   Total tokens: {len(encoding.ids)}")
    print(f"   First 15 tokens: {encoding.tokens[:15]}")
    
    # Example 4: Check special tokens
    print("\n4. Special token IDs:")
    special_tokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|pad|>"]
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        print(f"   {token}: {token_id}")
    
    # Example 5: Vocabulary size
    print(f"\n5. Vocabulary size: {tokenizer.get_vocab_size()}")
