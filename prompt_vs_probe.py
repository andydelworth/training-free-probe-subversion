import random
import argparse
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import shared components (no side-effects thanks to refactor in train_probe)
from train_probe import ProbeDataset, LinearProbe


def get_activation(model, tok, text):
    """Return last-token hidden state (same representation used during training)."""
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    return out["hidden_states"][-1][:, -1, :]  # shape: (1, hidden_dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="Monkeys")
    parser.add_argument("--val_pct", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_path", default=None, help="Path to saved probe .pt file")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Reproducible split
    # ------------------------------------------------------------------
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Load LLM (for fresh activations) & probe
    # ------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model_name)
    lm = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    probe = LinearProbe(input_dim=4096, output_dim=1)
    ckpt_path = args.ckpt_path or f"{args.data_name}_linear_probe.pt"
    probe.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    probe.eval()

    # ------------------------------------------------------------------
    # Load data strings & recreate validation IDs
    # ------------------------------------------------------------------
    data_file = os.path.join("data", args.data_name, "data.json")
    with open(data_file) as f:
        data = json.load(f)

    all_ids = list(range(len(data)))
    random.shuffle(all_ids)
    split_idx = int((1 - args.val_pct) * len(all_ids))
    test_ids = all_ids[split_idx:]

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    correct, total = 0, 0
    with torch.no_grad():
        for idx in test_ids:
            text = data[idx]["text"]
            label = torch.tensor(data[idx]["label"], dtype=torch.float32)
            act = get_activation(lm, tok, text).cpu()
            logit = probe(act).squeeze(1)
            pred = torch.round(torch.sigmoid(logit))
            correct += (pred == label).float().item()
            total += 1

    acc = correct / total if total > 0 else 0.0
    print(
        f"Accuracy on validation split with freshly-computed embeddings: {acc:.4f}"
    )


if __name__ == "__main__":
    main()

