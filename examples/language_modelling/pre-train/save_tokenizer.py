from tokenizers import Tokenizer
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = Tokenizer.from_pretrained(model_name)
pad_token_id = tokenizer.token_to_id("<|im_end|>")
tokenized = tokenizer.encode_batch(["Max Power"], add_special_tokens=False)

print(tokenized)
print(tokenized[0].ids)
print(tokenized[0].attention_mask)

print("Pad Token ID:", pad_token_id)
print("Pad Token:", tokenizer.decode([pad_token_id], False))
tokenizer.save("SmolLM2-135M-Instruct-Tokenizer.json")