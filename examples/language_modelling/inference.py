import torch
from tokenizers import Tokenizer

from wave_transformer.core.transformer import WaveTransformer
from wave_transformer.language_modelling.token_decoder import WaveToTokenDecoder
from wave_transformer.language_modelling.token_encoder import TokenToWaveEncoder
from wave_transformer.language_modelling.train_utils import generate_text, load_model_bundle, test_generation

# Load model
model = WaveTransformer.load(
    "./epoch_0_batch_4999",
    encoder_cls=TokenToWaveEncoder,
    decoder_cls=WaveToTokenDecoder,
    map_location=None
)
tokenizer = Tokenizer.from_file("pre-train/SmolLM2-135M-Instruct-Tokenizer.json")
pad_token_id = tokenizer.token_to_id("<|im_end|>") or 0

print("Pad Token ID:", pad_token_id)
print("Pad Token:", tokenizer.decode([pad_token_id], False))
test_generation(model=model.to(device="cuda"), tokenizer=tokenizer, device="cuda", repetition_penalty=1.2)
prompt = "All matter is composed of atoms"
gen = generate_text(model=model.to(device="cuda"), prompt=prompt, temperature=0.6, top_k=0, top_p=0.9, min_p=0.025, repetition_penalty=1.12, tokenizer=tokenizer, device="cuda")
print("Prompt:", prompt)
print("Generate:", gen)

# Diverse test prompts for early- to mid-training checkpoints.
# Runs each prompt with your generate_text() signature and prints outputs.

prompts = [
    # --- science / facts ---
    "All matter is composed of atoms",
    "Claim: The most informative and foundational concept in science,",
    "Define entropy in one clear sentence.",
    "Explain photosynthesis to a 10-year-old in 2 sentences.",
    "Summarize the periodic table in exactly 25 words.",

    # --- reasoning (final answer only) ---
    "Math puzzle: If a notebook costs $3 and a pen costs $2, how many notebooks can you buy with $17? Give the final answer only.",
    "Logic: If all mammals breathe air and whales are mammals, what do whales breathe? Answer in one word.",
    "You have 12 apples. You give away 5 and eat 2. How many are left? Final answer only.",

    # --- definitions / contrasts ---
    "Contrast: precision vs. recall—give 2 short bullet points.",
    "In 1–2 sentences, describe what a prime number is.",

    # --- style control / anti-boilerplate ---
    "Write a single paragraph about Daoism without any lists, references, or categories.",
    "Describe the moon in a lyrical style, 2 sentences, no quotations, no lists.",

    # --- instruction following / constraints ---
    "Give three safety rules for lab work as a numbered list of exactly 3 items.",
    "Write a haiku about winter (5-7-5 syllables).",
    "Explain gradient descent in exactly 3 sentences.",

    # --- world knowledge / geography ---
    "Name the capital of Japan and one famous landmark there. One sentence.",
    "What language is primarily spoken in Brazil? One word answer.",

    # --- programming text (no execution) ---
    "Write a Python docstring for a function `tokenize(text)` that returns a list of tokens.",
    "Give a short JSON example of a user profile with fields: name, age, interests (list).",
    "Explain what a hash map is in 2 concise sentences.",

    # --- analogy / metaphors ---
    "Give an analogy for how neural networks learn, in one sentence.",
    "Explain recursion to a child using a simple metaphor, one sentence.",

    # --- long-range dependency / coherence ---
    "Tell a tiny story (4–5 sentences) where a blue key introduced in sentence 1 is used to open a door in the last sentence.",
    "Write a paragraph that starts with: 'The tao that can be told' and stays philosophical, avoiding references and categories.",

    # --- counter-boilerplate endings ---
    "Write a short paragraph about butterflies. Do not include sections named 'References' or 'External links' or any categories.",

    # --- formatting / tables (simple text) ---
    "List 5 planets as a dash list (one per line).",
    "Provide a two-line CSV with columns city,country and rows for Paris/France and Nairobi/Kenya.",

    # --- typical sampling stress (creative but bounded) ---
    "Invent a new board game in 3 sentences, including its win condition.",
    "Describe a string with both ends fixed and how it oscillates, in one clear sentence.",

    # --- closing sanity ---
    "State one advantage of rotary position embeddings in transformers, one sentence.",
]

for prompt in prompts:
    gen = generate_text(
        model=model.to(device="cuda"),
        prompt=prompt,
        temperature=0.6,
        top_k=0,
        top_p=0.9,
        min_p=0.025,              # typical sampling floor
        repetition_penalty=1.12,
        tokenizer=tokenizer,
        device="cuda",
        # optionally: max_new_tokens=120,
    )
    print("Prompt:", prompt)
    print("Generate:", gen)
    print("-" * 80)


