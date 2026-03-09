from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from huggingface_hub import hf_hub_download, notebook_login
import numpy as np
import einops
import textwrap
from typing import Literal
import plotly.express as px
from functools import partial
import dataclasses
import gc
import pandas as pd
from safetensors.torch import load_file
import torch
import torch.nn as nn

torch.set_grad_enabled(False)

model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-pt",
        device_map='auto',
        )
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
