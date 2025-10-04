import copy

from tokenizers import processors

from tokenizers import Tokenizer
model_name = "SmolLM2-135M-Instruct-Tokenizer.json"
train_tokenizer = Tokenizer.from_file(model_name)
seq_len = 128
train_tokenizer.add_special_tokens(["<|bos|>", "<|eos|>", "<|pad|>"])

bos_token_id = train_tokenizer.token_to_id("<|bos|>")
eos_token_id = train_tokenizer.token_to_id("<|eos|>")
pad_token_id = train_tokenizer.token_to_id("<|pad|>")

tokenizer = copy.deepcopy(train_tokenizer)
train_tokenizer.post_processor = processors.TemplateProcessing(
    single="<|bos|> $A <|eos|>",
    pair="<|bos|> $A <|eos|> <|bos|> $B <|eos|>",
    special_tokens=[
        ("<|bos|>", bos_token_id),
        ("<|eos|>", eos_token_id),
    ],
)
tokenizer.post_processor = processors.TemplateProcessing(
    single="<|bos|> $A",
    pair="<|bos|> $A <|bos|> $B",
    special_tokens=[
        ("<|bos|>", bos_token_id)
    ],
)
train_tokenizer.enable_padding(pad_id=pad_token_id, pad_token="<|pad|>", length=seq_len)
train_tokenizer.enable_truncation(max_length=seq_len - 2)
print("Bos Token ID:", bos_token_id)
print("Bos Token:", tokenizer.decode([bos_token_id], False))
print("Eos Token ID:", eos_token_id)
print("Eos Token:", tokenizer.decode([eos_token_id], False))
print("Pad Token ID:", pad_token_id)
print("Pad Token:", tokenizer.decode([pad_token_id], False))

prompt = "Hello, World!"
tokenized = train_tokenizer.encode(prompt)
print(prompt)
print(tokenized.ids)
print(len(tokenized.ids))

tokenized = tokenizer.encode(prompt)
print(prompt)
print(tokenized.ids)
print(len(tokenized.ids))

tokenizer.save("tokenizer-truncating_padding_bos_eos.json")