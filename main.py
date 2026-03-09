from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
import torch.nn as nn
import math

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"


class JumpReluSAE(nn.Module):
    def __init__(self, w_enc, b_enc, threshold, w_dec, b_dec):
        super().__init__()
        self.w_enc = nn.Parameter(w_enc)
        self.b_enc = nn.Parameter(b_enc)
        self.threshold = nn.Parameter(threshold)
        self.w_dec = nn.Parameter(w_dec)
        self.b_dec = nn.Parameter(b_dec)

    def encode(self, x):
        pre_acts = x @ self.w_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.relu(pre_acts)
        return acts


def load_sae(layer=22, width="65k", l0="medium", category="resid_post", device=device) -> JumpReluSAE:
    path = f"{category}/layer_{layer}_width_{width}_l0_{l0}/params.safetensors"
    local_path = hf_hub_download(repo_id="google/gemma-scope-2-1b-pt", filename=path)
    tensors = load_file(local_path)
    sae = JumpReluSAE(
        w_enc=tensors["w_enc"],
        b_enc=tensors["b_enc"],
        threshold=tensors["threshold"],
        w_dec=tensors["w_dec"],
        b_dec=tensors["b_dec"],
    )
    return sae.to(device).eval()


def get_residual_stream(model, layer_idx, input_ids) -> torch.Tensor:
    captured = []

    def hook_fn(_module, _input, output):
        captured.append(output[0].detach())
        hook.remove()

    hook = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    model(input_ids)
    return captured[0].squeeze(0)  # (seq_len, d_model)


def inspect_live(prompt, model, tokenizer, sae, layer, top_k=10):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    residual = get_residual_stream(model, layer, input_ids)   # (seq_len, d_model)
    sae_acts = sae.encode(residual.float())                   # (seq_len, d_sae)

    tokens_nobos = tokens[1:]
    acts_nobos = sae_acts[1:]                                 # (seq_len-1, d_sae)

    mean_acts = acts_nobos.mean(dim=0)                        # (d_sae,)
    topk_indices = mean_acts.topk(top_k).indices              # (top_k,)

    print(f"\nPrompt: {prompt!r}")
    print(f"\n=== Top-{top_k} Active Features (layer {layer}) ===")
    for rank, feat_idx in enumerate(topk_indices.tolist(), start=1):
        mean_val = mean_acts[feat_idx].item()
        print(f"  #{rank:<3} Feature {feat_idx:>6}  mean_act={mean_val:.4f}")

    print()
    seq_len = acts_nobos.shape[0]
    n_top_tokens = max(1, math.ceil(0.1 * seq_len))

    for feat_idx in topk_indices.tolist():
        feat_acts = acts_nobos[:, feat_idx]  # (seq_len-1,)
        if feat_acts.max().item() == 0:
            continue
        top_token_indices = feat_acts.topk(n_top_tokens).indices.tolist()
        print(f"Feature {feat_idx}:")
        for tok_idx in top_token_indices:
            tok = tokens_nobos[tok_idx]
            val = feat_acts[tok_idx].item()
            print(f'    \u25b8 {tok!r:<12}: {val:.4f}')

    l0_per_token = (acts_nobos > 0).sum(dim=1).tolist()
    print(f"\n=== L0 per token (excluding BOS) ===")
    for tok, l0 in zip(tokens_nobos, l0_per_token):
        print(f"    {tok!r:<16}: {l0}")

    avg_l0 = sum(l0_per_token) / len(l0_per_token)
    print(f"\nAverage L0 (non-BOS): {avg_l0:.2f}")


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-pt",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    sae = load_sae(layer=22, width="65k", l0="medium", device=device)

    prompt = "The law of conservation of energy states that energy cannot be created or destroyed."
    inspect_live(prompt, model, tokenizer, sae, layer=22, top_k=10)
