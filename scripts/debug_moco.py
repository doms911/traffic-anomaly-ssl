"""Debug MoCo: inspect logits and loss components in one forward pass."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.cityscapes_ssl import CityscapesSSL
from src.datasets.ssl_transforms import build_moco_v2_transforms
from src.models.moco import MoCo
from src.paths import CITYSCAPES_ROOT


def main():
    device = torch.device("cuda")
    model = MoCo(queue_size=4096, temperature=0.2).to(device)
    # Učitaj trenutni checkpoint
    ckpt = torch.load("checkpoints/moco_smoke/epoch_1.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print("Loaded epoch_1 checkpoint")

    transform = build_moco_v2_transforms(crop_size=224)
    ds = CityscapesSSL(
        root=str(CITYSCAPES_ROOT), split="train", mode="image", transform=transform
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)

    view_q, view_k = next(iter(loader))
    view_q = view_q.to(device)
    view_k = view_k.to(device)

    model.eval()  # no dropout/BN running stats update

    # Manually compute q, k (skip momentum update and queue)
    q = model.encoder_q(view_q)
    k = model.encoder_k(view_k)

    print(f"q shape: {q.shape}, norm range: [{q.norm(dim=1).min():.3f}, {q.norm(dim=1).max():.3f}]")
    print(f"k shape: {k.shape}, norm range: [{k.norm(dim=1).min():.3f}, {k.norm(dim=1).max():.3f}]")
    print(f"(Norms should all be ~1.0 since we L2-normalize)\n")

    # Pos logits
    l_pos = torch.einsum("nc,nc->n", q, k)
    print(f"l_pos (q·k for matched pairs):")
    print(f"  mean: {l_pos.mean():.4f}, std: {l_pos.std():.4f}")
    print(f"  range: [{l_pos.min():.4f}, {l_pos.max():.4f}]")
    print(f"  (Random init: should be high ~0.9 since q and k encoders identical at init)\n")

    # Neg logits vs queue
    l_neg = torch.einsum("nc,ck->nk", q, model.queue.clone().detach())
    print(f"l_neg (q·queue):")
    print(f"  mean: {l_neg.mean():.4f}, std: {l_neg.std():.4f}")
    print(f"  range: [{l_neg.min():.4f}, {l_neg.max():.4f}]")
    print(f"  (Random queue: should be ~0 mean, small std)\n")

    # Concatenate and apply temperature
    logits = torch.cat([l_pos.unsqueeze(-1), l_neg], dim=1) / 0.2
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

    print(f"logits[0, :5]: {logits[0, :5].tolist()}")
    print(f"  (logits[0,0] is positive, [0,1:] are negatives)")
    print(f"  Positive logit: {logits[0, 0].item():.4f}")
    print(f"  Max negative logit: {logits[0, 1:].max().item():.4f}")
    print(f"  Argmax: {logits[0].argmax().item()} (should be 0 if positive wins)\n")

    loss = F.cross_entropy(logits, labels)
    pred = logits.argmax(dim=1)
    acc = (pred == labels).float().mean()
    print(f"Loss: {loss.item():.4f}")
    print(f"Accuracy: {acc.item()*100:.2f}%")
    print(f"How many predicted as positive (class 0): {(pred == 0).sum().item()}/{len(pred)}")


if __name__ == "__main__":
    main()