import copy

import torch
from tokenizers import Tokenizer, processors
from tokenizers.processors import TemplateProcessing

from wave_transformer.core.transformer import WaveTransformer
from wave_transformer.language_modelling.token_decoder import WaveToTokenDecoder
from wave_transformer.language_modelling.token_encoder import TokenToWaveEncoder, TokenToWaveEncoderSlim
from wave_transformer.language_modelling.train_utils import generate_text, load_model_bundle, test_generation

# Load model
model = WaveTransformer.load(
    "./results_wikitext_v1_raw/epoch_0_batch_19999",
    encoder_cls=TokenToWaveEncoderSlim,
    decoder_cls=WaveToTokenDecoder,
    map_location=None
)
from tokenizers import Tokenizer
model_name = "./pre-train/SmolLM2-135M-Instruct-Tokenizer.json"
tokenizer = Tokenizer.from_file(model_name)
seq_len = 128
tokenizer.add_special_tokens(["<|bos|>", "<|eos|>", "<|pad|>"])

bos_token_id = tokenizer.token_to_id("<|bos|>")
eos_token_id = tokenizer.token_to_id("<|eos|>")
pad_token_id = tokenizer.token_to_id("<|pad|>")

tokenizer.post_processor = processors.TemplateProcessing(
    single="<|bos|> $A",
    pair="<|bos|> $A <|bos|> $B",
    special_tokens=[
        ("<|bos|>", bos_token_id)
    ],
)
#test_generation(model=model.to(device="cuda"), tokenizer=tokenizer, device="cuda", repetition_penalty=1.2)
#prompt = "All matter is composed of atoms"
#gen = generate_text(model=model.to(device="cuda"), prompt=prompt, temperature=0.6, top_k=0, top_p=0.9, min_p=0.025, repetition_penalty=1.12, tokenizer=tokenizer, device="cuda")
#print("Prompt:", prompt)
#print("Generate:", gen)

# Diverse test prompts for early- to mid-training checkpoints.
# Runs each prompt with your generate_text() signature and prints outputs.

prompts = [
    # Geographic/Location prompts
    "The Amazon River is located in",
    "Tokyo is the capital city of",
    "The Sahara Desert covers parts of",
    "Mount Everest is situated in the",
    "The Mediterranean Sea is bordered by",

    # Historical/Temporal prompts
    "World War II began in the year",
    "The Roman Empire fell in",
    "The Renaissance period started in",
    "The first human moon landing occurred in",
    "The American Revolutionary War took place during",

    # Taxonomic/Scientific prompts
    "Homo sapiens is a species in the family",
    "The genus Felis includes",
    "Photosynthesis is a process by which",
    "DNA stands for",
    "The periodic table organizes elements by",

    # Definitional prompts
    "A democracy is a form of government in which",
    "An atom consists of",
    "The speed of light in a vacuum is",
    "Gravity is a fundamental force that",
    "A mammal is defined as an animal that",

    # Biographical prompts
    "Albert Einstein was a physicist known for",
    "William Shakespeare was an English playwright who",
    "Leonardo da Vinci was an Italian polymath who",
    "Marie Curie was a scientist who discovered",

    # Article continuation prompts
    "Paris is the capital and most populous city of France.",
    "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
    "Carbon is a chemical element with the symbol C and atomic number 6.",
    "The Internet is a global system of interconnected computer networks that",
    "Beethoven's Symphony No. 9 was composed between",
]

for prompt in prompts:
    gen = generate_text(
        model=model.to(device="cuda"),
        prompt=prompt,
        temperature=0.65,
        top_k=0,
        top_p=0.9,
        min_p=0.025,              # typical sampling floor
        repetition_penalty=1.10,
        max_tokens=50,
        tokenizer=tokenizer,
        device="cuda",
        # optionally: max_new_tokens=120,
    )
    print("Prompt:", prompt)
    print("Generate:", gen)
    print("-" * 80)


