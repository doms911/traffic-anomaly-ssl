"""Test that momentum update actually moves key encoder when query encoder changes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from src.models.moco import MoCo


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoCo(queue_size=1024).to(device)
    optimizer = torch.optim.SGD(model.encoder_q.parameters(), lr=0.1)

    q_param_before = next(model.encoder_q.parameters()).clone()
    k_param_before = next(model.encoder_k.parameters()).clone()

    print(f"Initial diff q-k: {(q_param_before - k_param_before).abs().mean():.6f}")
    print("(Expected 0 - we copied q -> k in __init__)\n")

    # Do a few real training steps
    for step in range(5):
        view_q = torch.randn(4, 3, 224, 224, device=device)
        view_k = torch.randn(4, 3, 224, 224, device=device)

        logits, labels = model(view_q, view_k)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        q_param = next(model.encoder_q.parameters())
        k_param = next(model.encoder_k.parameters())

        q_moved = (q_param - q_param_before).abs().mean().item()
        k_moved = (k_param - k_param_before).abs().mean().item()
        qk_diff = (q_param - k_param).abs().mean().item()

        print(f"Step {step+1}: q moved {q_moved:.6f}, k moved {k_moved:.6f}, q-k diff {qk_diff:.6f}")

    print("\nExpected: q moves a lot (gradient updates), k moves a tiny bit (momentum 0.999)")
    print("q-k diff should grow slowly, not stay 0.")


if __name__ == "__main__":
    main()