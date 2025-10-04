import copy

from datasets import load_dataset
from tokenizers import Tokenizer, processors

from wave_transformer.language_modelling.streaming_dataset import BoundedStreamingDataset, MultiBoundedStreamingDataset
print("Loading WikiText-103...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

seq_len = 512
model_name = "SmolLM2-135M-Instruct-Tokenizer.json"
train_tokenizer = Tokenizer.from_file(model_name)

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


vocab_size = train_tokenizer.get_vocab_size()
avg_train_length = 0
max_train_length = -1
min_train_length = 10000

train_corpus_size = 0
for sample in dataset["train"]:
    train_length = len(tokenizer.encode(sample["text"]).ids)
    avg_train_length += train_length
    max_train_length = max(max_train_length, train_length)
    min_train_length = min(min_train_length, train_length)
    train_corpus_size += 1
entries_per_dataset = train_corpus_size
avg_train_length = avg_train_length / train_corpus_size

print(
        f"Dataset Entries: {entries_per_dataset}, Average length: {avg_train_length}, Max length: {max_train_length}, Min length: {min_train_length}")