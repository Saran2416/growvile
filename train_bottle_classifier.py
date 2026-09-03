"""Train a lightweight bottle subtype classifier from ImageFolder data."""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="ImageFolder root with one directory per subtype")
    parser.add_argument("--output", default="bottle_subtypes.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    weights = MobileNet_V2_Weights.DEFAULT
    dataset = datasets.ImageFolder(args.data, transform=weights.transforms())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, len(dataset.classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(args.epochs):
        for images, labels in loader:
            optimizer.zero_grad()
            loss_fn(model(images), labels).backward()
            optimizer.step()
    torch.save({"model": model.state_dict(), "classes": dataset.classes}, args.output)
    print(f"saved {args.output}; classes={dataset.classes}")


if __name__ == "__main__":
    main()