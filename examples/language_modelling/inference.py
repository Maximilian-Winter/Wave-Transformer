import torch

from wave_transformer.language_modelling.train_utils import generate_text


model_checkpoint = torch.load("./wave_transformer_checkpoint")["model_state_dict"]

