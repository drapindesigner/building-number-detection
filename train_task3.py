# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Author: Mohammad Saif Ul Haq
# Last Modified: 2025-10-03

"""Train the Task 3 character recogniser using EMNIST and optional auxiliary datasets."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2 as cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets

from perception.config import RecognitionConfig
from perception.recognition import CharacterCNN
from perception.recognition.preprocess import preprocess_character


@dataclass
class TrainingConfig:
    data_root: Path
    output_path: Path
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    charset: str
    device: str
    custom_data: Path | None
    custom_repeat: int
    use_svhn: bool
    use_usps: bool


def _build_class_map(dataset: datasets.EMNIST, charset: str) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    allowed = {ch: idx for idx, ch in enumerate(charset)}
    for cls_index, label in enumerate(dataset.classes):
        if isinstance(label, bytes):
            label = label.decode("utf-8")
        character = str(label).strip().upper()
        if len(character) != 1:
            continue
        target = allowed.get(character)
        if target is not None:
            mapping[cls_index] = target
    return mapping


class _FilteredEMNIST(Dataset):
    def __init__(self, dataset: datasets.EMNIST, class_map: Dict[int, int], image_size: int) -> None:
        self.dataset = dataset
        self.indices: List[int] = []
        mapped_labels: List[int] = []
        for idx, target in enumerate(dataset.targets):
            label = int(target)
            mapped = class_map.get(label)
            if mapped is not None:
                self.indices.append(idx)
                mapped_labels.append(mapped)
        self.mapped_labels = torch.tensor(mapped_labels, dtype=torch.long)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, _ = self.dataset[self.indices[idx]]
        if isinstance(image, torch.Tensor):
            array = image.squeeze(0).numpy()
            array = (array * 255.0).astype(np.uint8)
        else:
            array = np.array(image)
        array = np.fliplr(np.rot90(array, k=1))
        processed = preprocess_character(array, self.image_size)
        tensor = torch.from_numpy(processed).unsqueeze(0)
        return tensor, int(self.mapped_labels[idx].item())


class CustomCharacterDataset(Dataset):
    def __init__(self, root: Path, charset: str, image_size: int, augment: bool = True) -> None:
        self.samples: List[Tuple[Path, int]] = []
        mapping = {char: idx for idx, char in enumerate(charset)}
        extensions = ("*.png", "*.jpg", "*.jpeg")
        for subdir in sorted(root.iterdir()):
            if not subdir.is_dir():
                continue
            char = subdir.name.strip().upper()
            target = mapping.get(char)
            if target is None:
                continue
            for pattern in extensions:
                for path in sorted(subdir.glob(pattern)):
                    self.samples.append((path, target))
        self.image_size = image_size
        self.augment = augment
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        if self.augment:
            image = self._augment(image)
        processed = preprocess_character(image, self.image_size)
        tensor = torch.from_numpy(processed).unsqueeze(0)
        return tensor, label

    def _augment(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        angle = float(self._rng.uniform(-8.0, 8.0))
        scale = float(self._rng.uniform(0.85, 1.15))
        tx = float(self._rng.uniform(-0.08, 0.08) * w)
        ty = float(self._rng.uniform(-0.08, 0.08) * h)
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty
        warped = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        if self._rng.random() < 0.35:
            warped = cv2.bitwise_not(warped)
        alpha = float(self._rng.uniform(0.75, 1.35))
        beta = float(self._rng.uniform(-35, 35))
        adjusted = cv2.convertScaleAbs(warped, alpha=alpha, beta=beta)
        if self._rng.random() < 0.4:
            ksize = int(self._rng.integers(0, 2) * 2 + 1)
            if ksize > 1:
                adjusted = cv2.GaussianBlur(adjusted, (ksize, ksize), 0)
        if self._rng.random() < 0.3:
            kernel_size = int(self._rng.integers(1, 3))
            if kernel_size > 0:
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                if self._rng.random() < 0.5:
                    adjusted = cv2.dilate(adjusted, kernel, iterations=1)
                else:
                    adjusted = cv2.erode(adjusted, kernel, iterations=1)
        noise_sigma = float(self._rng.uniform(0.0, 12.0))
        if noise_sigma > 0.0:
            noise = self._rng.normal(0.0, noise_sigma, size=adjusted.shape)
            adjusted = np.clip(adjusted.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
        return adjusted


class SVHNDataset(Dataset):
    def __init__(self, root: Path, split: str, charset: str, image_size: int) -> None:
        self.dataset = datasets.SVHN(root=str(root), split=split, download=True)
        self.image_size = image_size
        mapping = {char: idx for idx, char in enumerate(charset)}
        self.samples: List[Tuple[int, int]] = []
        labels_source = getattr(self.dataset, "labels", None)
        if labels_source is None:
            labels_source = getattr(self.dataset, "y", None)
        if labels_source is None:
            labels_source = [self.dataset[idx][1] for idx in range(len(self.dataset))]
        for idx, label in enumerate(labels_source):
            digit = int(label)
            if digit == 10:
                digit = 0
            char = str(digit)
            target = mapping.get(char)
            if target is not None:
                self.samples.append((idx, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample_idx, label = self.samples[idx]
        image, _ = self.dataset[sample_idx]
        if isinstance(image, torch.Tensor):
            array = image.numpy()
        else:
            array = np.array(image)
        if array.ndim == 3 and array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))
        if array.ndim == 3:
            gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        else:
            gray = array.astype(np.uint8)
        processed = preprocess_character(gray, self.image_size)
        tensor = torch.from_numpy(processed).unsqueeze(0)
        return tensor, label


class USPSDataset(Dataset):
    def __init__(self, root: Path, train: bool, charset: str, image_size: int) -> None:
        self.dataset = datasets.USPS(root=str(root), train=train, download=True)
        self.image_size = image_size
        mapping = {char: idx for idx, char in enumerate(charset)}
        self.samples: List[Tuple[int, int]] = []
        targets = self.dataset.targets
        for idx, label in enumerate(targets):
            digit = int(label)
            char = str(digit)
            target = mapping.get(char)
            if target is not None:
                self.samples.append((idx, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample_idx, label = self.samples[idx]
        image, _ = self.dataset[sample_idx]
        if isinstance(image, torch.Tensor):
            array = image.squeeze(0).numpy() * 255.0
            array = array.astype(np.uint8)
        else:
            array = np.array(image)
        processed = preprocess_character(array, self.image_size)
        tensor = torch.from_numpy(processed).unsqueeze(0)
        return tensor, label


def _num_workers() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def _make_dataloaders(cfg: TrainingConfig, recognition_cfg: RecognitionConfig) -> Tuple[DataLoader, DataLoader, str]:
    train_dataset = datasets.EMNIST(root=str(cfg.data_root), split="balanced", train=True, download=True, transform=None)
    test_dataset = datasets.EMNIST(root=str(cfg.data_root), split="balanced", train=False, download=True, transform=None)
    class_map = _build_class_map(train_dataset, recognition_cfg.charset)
    train_filtered = _FilteredEMNIST(train_dataset, class_map, recognition_cfg.image_size)
    test_filtered = _FilteredEMNIST(test_dataset, class_map, recognition_cfg.image_size)
    train_sources: List[Dataset] = [train_filtered]
    if cfg.use_svhn:
        train_sources.append(SVHNDataset(cfg.data_root / "svhn", "train", recognition_cfg.charset, recognition_cfg.image_size))
    if cfg.use_usps:
        train_sources.append(USPSDataset(cfg.data_root / "usps", True, recognition_cfg.charset, recognition_cfg.image_size))
    if cfg.custom_data and cfg.custom_data.exists():
        custom_dataset = CustomCharacterDataset(cfg.custom_data, recognition_cfg.charset, recognition_cfg.image_size)
        if len(custom_dataset) > 0:
            repeats = max(1, int(cfg.custom_repeat))
            for _ in range(repeats):
                train_sources.append(custom_dataset)
    if len(train_sources) == 1:
        train_combined: Dataset = train_sources[0]
    else:
        train_combined = ConcatDataset(train_sources)
    workers = _num_workers()
    train_loader = DataLoader(train_combined, batch_size=cfg.batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_filtered, batch_size=cfg.batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader, recognition_cfg.charset


def train(cfg: TrainingConfig) -> None:
    recognition_cfg = RecognitionConfig(
        weights_path=cfg.output_path,
        device=cfg.device,
        charset=cfg.charset,
    )
    train_loader, test_loader, charset = _make_dataloaders(cfg, recognition_cfg)

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CharacterCNN(num_classes=len(charset)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_acc = 0.0
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / max(1, val_total)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg.output_path)
            print(f"Saved checkpoint to {cfg.output_path}")

    print(f"Training complete. Best validation accuracy: {best_acc:.3f}")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train character recognition CNN (Task 3)")
    parser.add_argument("--data-root", type=Path, default=Path("data/emnist"), help="Folder to store EMNIST data")
    parser.add_argument("--output", type=Path, default=Path("data/recognizer/best.pt"), help="Where to store the trained weights")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--charset", type=str, default="0123456789ABCDEFGHIJKLMNOP", help="Target character set")
    parser.add_argument("--device", type=str, default="", help="Device to train on (e.g. 'cuda' or 'cpu')")
    parser.add_argument("--custom-data", type=Path, default=None, help="Optional directory of labelled characters for fine-tuning")
    parser.add_argument(
        "--custom-repeat",
        type=int,
        default=4,
        help="Number of times to repeat the custom dataset when mixing with EMNIST",
    )
    parser.add_argument("--use-svhn", action="store_true", help="Include SVHN dataset")
    parser.add_argument("--use-usps", action="store_true", help="Include USPS dataset")
    args = parser.parse_args()
    return TrainingConfig(
        data_root=args.data_root,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        charset=args.charset,
        device=args.device,
        custom_data=args.custom_data,
        custom_repeat=args.custom_repeat,
        use_svhn=args.use_svhn,
        use_usps=args.use_usps,
    )


if __name__ == "__main__":
    train(parse_args())
